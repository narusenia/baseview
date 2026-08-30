use std::ffi::c_void;
use std::str::FromStr;

use raw_window_handle::RawWindowHandle;

use cocoa::appkit::{
    NSOpenGLContext, NSOpenGLContextParameter, NSOpenGLPFAAccelerated, NSOpenGLPFAAlphaSize,
    NSOpenGLPFAColorSize, NSOpenGLPFADepthSize, NSOpenGLPFADoubleBuffer, NSOpenGLPFAMultisample,
    NSOpenGLPFAOpenGLProfile, NSOpenGLPFASampleBuffers, NSOpenGLPFASamples, NSOpenGLPFAStencilSize,
    NSOpenGLPixelFormat, NSOpenGLProfileVersion3_2Core, NSOpenGLProfileVersion4_1Core,
    NSOpenGLProfileVersionLegacy, NSView,
};
use cocoa::base::{id, nil, YES};
use cocoa::foundation::{NSPoint, NSRect, NSSize, NSString};

use core_foundation::base::TCFType;
use core_foundation::bundle::{CFBundleGetBundleWithIdentifier, CFBundleGetFunctionPointerForName};
use core_foundation::string::CFString;

use objc::{class, msg_send, sel, sel_impl};

use super::{GlConfig, GlError, Profile};

pub type CreationFailedError = ();

// The GL names this backend needs. Declared here rather than pulled from a
// loader: the OpenGL framework is linked already, and these are the handful of
// entry points the surface path uses.
#[link(name = "OpenGL", kind = "framework")]
extern "C" {
    fn glGenFramebuffers(n: i32, ids: *mut u32);
    fn glDeleteFramebuffers(n: i32, ids: *const u32);
    fn glBindFramebuffer(target: u32, id: u32);
    fn glFramebufferTexture2D(target: u32, attachment: u32, textarget: u32, texture: u32, level: i32);
    fn glFramebufferRenderbuffer(target: u32, attachment: u32, rbtarget: u32, renderbuffer: u32);
    fn glGenTextures(n: i32, ids: *mut u32);
    fn glDeleteTextures(n: i32, ids: *const u32);
    fn glBindTexture(target: u32, id: u32);
    fn glTexParameteri(target: u32, pname: u32, param: i32);
    fn glGenRenderbuffers(n: i32, ids: *mut u32);
    fn glDeleteRenderbuffers(n: i32, ids: *const u32);
    fn glBindRenderbuffer(target: u32, id: u32);
    fn glRenderbufferStorage(target: u32, format: u32, width: i32, height: i32);
    fn glCheckFramebufferStatus(target: u32) -> u32;
    fn glBlitFramebuffer(
        sx0: i32, sy0: i32, sx1: i32, sy1: i32,
        dx0: i32, dy0: i32, dx1: i32, dy1: i32,
        mask: u32, filter: u32,
    );
    fn glFlush();

    fn CGLTexImageIOSurface2D(
        ctx: *mut c_void,
        target: u32,
        internal_format: u32,
        width: u32,
        height: u32,
        format: u32,
        ty: u32,
        io_surface: *mut c_void,
        plane: u32,
    ) -> i32;
}

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: id) -> *mut c_void;
}

const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_DEPTH_STENCIL_ATTACHMENT: u32 = 0x821A;
const GL_RENDERBUFFER: u32 = 0x8D41;
const GL_DEPTH24_STENCIL8: u32 = 0x88F0;
const GL_TEXTURE_RECTANGLE: u32 = 0x84F5;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_LINEAR: i32 = 0x2601;
const GL_RGBA: u32 = 0x1908;
const GL_BGRA: u32 = 0x80E1;
const GL_UNSIGNED_INT_8_8_8_8_REV: u32 = 0x8367;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
const GL_READ_FRAMEBUFFER: u32 = 0x8CA8;
const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
const GL_COLOR_BUFFER_BIT: u32 = 0x4000;
const GL_NEAREST: u32 = 0x2600;
const GL_RGBA8: u32 = 0x8058;

/// An `IOSurface` to show, and the buffers a frame is built in before it gets
/// there.
///
/// **The frame is not drawn straight into the surface.** OpenGL puts its first
/// row at the bottom and CoreAnimation reads a surface's first row as the top,
/// so a frame drawn directly into it arrives upside down — and neither
/// `geometryFlipped` nor a layer transform turns the *contents* back over.
/// Drawing into ordinary buffers and blitting into the surface with the
/// destination's Y reversed does, in one GPU copy of the window.
struct Surface {
    io_surface: *mut c_void,
    io_texture: u32,
    io_framebuffer: u32,
    colour: u32,
    depth_stencil: u32,
    width: u32,
    height: u32,
}

/// An OpenGL context that draws into an `IOSurface`, which a plain `CALayer`
/// then shows.
///
/// **There is no `NSOpenGLView`, and that is the whole point.** An
/// `NSOpenGLView` added into a host's window drops that window onto a
/// compatibility compositing path it never leaves: the host stays sluggish
/// until it is relaunched, long after the plugin has been removed. Confirmed in
/// Studio One (badly) and Ableton Live (mildly) on 2026-08-30 — never opening
/// the editor keeps the host fine, and one open is enough to spoil it. Other
/// plugins do not do this because most of them do not put an OpenGL surface in
/// the host's window.
///
/// What goes into the host's window here is an ordinary layer-backed `NSView`
/// whose layer's `contents` is an `IOSurface`, which is the same thing any
/// other layer is. The drawing is unchanged: the caller renders into
/// [`Self::framebuffer`] exactly as it rendered into the window before.
pub struct GlContext {
    /// The layer that shows the surface. A sublayer of the caller's own view's
    /// layer, so it draws in the right place and catches no input.
    layer: id,
    context: id,
    /// The framebuffer the caller draws into. **Its name never changes**, so a
    /// renderer told about it once stays correct across a resize — only the
    /// attachments are replaced.
    framebuffer: u32,
    surface: std::cell::RefCell<Surface>,
    scale: std::cell::Cell<f64>,
}

impl Surface {
    /// Makes an `IOSurface` and points a rectangle texture at it.
    ///
    /// **`BGRA` and `GL_TEXTURE_RECTANGLE`** are what `CGLTexImageIOSurface2D`
    /// accepts; neither is a preference.
    unsafe fn new(cgl: *mut c_void, width: u32, height: u32) -> Result<Surface, GlError> {
        let width = width.max(1);
        let height = height.max(1);

        // Built by hand rather than through `NSDictionary`'s variadic
        // constructor, which this binding types differently from what the keys
        // need here.
        let properties: id = msg_send![class!(NSMutableDictionary), dictionary];
        let mut put = |key: &str, value: i64| {
            let key = NSString::alloc(nil).init_str(key);
            let () = msg_send![properties, setObject: number(value) forKey: key];
        };
        put("IOSurfaceWidth", i64::from(width));
        put("IOSurfaceHeight", i64::from(height));
        put("IOSurfaceBytesPerElement", 4);
        put("IOSurfacePixelFormat", i64::from(u32::from_be_bytes(*b"BGRA")));

        let io_surface = IOSurfaceCreate(properties);
        if io_surface.is_null() {
            return Err(GlError::CreationFailed(()));
        }

        let mut io_texture = 0;
        glGenTextures(1, &mut io_texture);
        glBindTexture(GL_TEXTURE_RECTANGLE, io_texture);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        let result = CGLTexImageIOSurface2D(
            cgl,
            GL_TEXTURE_RECTANGLE,
            GL_RGBA,
            width,
            height,
            GL_BGRA,
            GL_UNSIGNED_INT_8_8_8_8_REV,
            io_surface,
            0,
        );
        glBindTexture(GL_TEXTURE_RECTANGLE, 0);
        if result != 0 {
            return Err(GlError::CreationFailed(()));
        }

        let mut io_framebuffer = 0;
        glGenFramebuffers(1, &mut io_framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, io_framebuffer);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_RECTANGLE,
            io_texture,
            0,
        );
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE {
            return Err(GlError::CreationFailed(()));
        }

        // What the frame is actually drawn into. femtovg's `set_screen_target`
        // requires depth and stencil, and vizia clips with the stencil buffer.
        let mut colour = 0;
        glGenRenderbuffers(1, &mut colour);
        glBindRenderbuffer(GL_RENDERBUFFER, colour);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width as i32, height as i32);

        let mut depth_stencil = 0;
        glGenRenderbuffers(1, &mut depth_stencil);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_stencil);
        glRenderbufferStorage(
            GL_RENDERBUFFER,
            GL_DEPTH24_STENCIL8,
            width as i32,
            height as i32,
        );
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        Ok(Surface { io_surface, io_texture, io_framebuffer, colour, depth_stencil, width, height })
    }

    unsafe fn attach_to(&self, framebuffer: u32) -> Result<(), GlError> {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_RENDERBUFFER,
            self.colour,
        );
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_STENCIL_ATTACHMENT,
            GL_RENDERBUFFER,
            self.depth_stencil,
        );
        let status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if status != GL_FRAMEBUFFER_COMPLETE {
            return Err(GlError::CreationFailed(()));
        }
        Ok(())
    }

    /// Copies the finished frame into the surface, **turning it over on the
    /// way** — the destination's Y runs from `height` to `0`.
    unsafe fn publish(&self, framebuffer: u32) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.io_framebuffer);
        glBlitFramebuffer(
            0,
            0,
            self.width as i32,
            self.height as i32,
            0,
            self.height as i32,
            self.width as i32,
            0,
            GL_COLOR_BUFFER_BIT,
            GL_NEAREST,
        );
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    }

    unsafe fn destroy(&self) {
        glDeleteFramebuffers(1, &self.io_framebuffer);
        glDeleteTextures(1, &self.io_texture);
        glDeleteRenderbuffers(1, &self.colour);
        glDeleteRenderbuffers(1, &self.depth_stencil);
        let () = msg_send![self.io_surface as id, release];
    }
}

/// An action dictionary that disables every implicit animation.
unsafe fn null_actions() -> id {
    let null: id = msg_send![class!(NSNull), null];
    let actions: id = msg_send![class!(NSMutableDictionary), dictionary];
    for key in ["contents", "position", "bounds", "frame", "transform"] {
        let key = NSString::alloc(nil).init_str(key);
        let () = msg_send![actions, setObject: null forKey: key];
    }
    actions
}

unsafe fn number(value: i64) -> id {
    let class = objc::runtime::Class::get("NSNumber").unwrap();
    msg_send![class, numberWithLongLong: value]
}

impl GlContext {
    pub unsafe fn create(parent: &RawWindowHandle, config: GlConfig) -> Result<GlContext, GlError> {
        let handle = if let RawWindowHandle::AppKit(handle) = parent {
            handle
        } else {
            return Err(GlError::InvalidWindowHandle);
        };

        if handle.ns_view.is_null() {
            return Err(GlError::InvalidWindowHandle);
        }

        let parent_view = handle.ns_view as id;

        let version = if config.version < (3, 2) && config.profile == Profile::Compatibility {
            NSOpenGLProfileVersionLegacy
        } else if config.version == (3, 2) && config.profile == Profile::Core {
            NSOpenGLProfileVersion3_2Core
        } else if config.version > (3, 2) && config.profile == Profile::Core {
            NSOpenGLProfileVersion4_1Core
        } else {
            return Err(GlError::VersionNotSupported);
        };

        #[rustfmt::skip]
        let mut attrs = vec![
            NSOpenGLPFAOpenGLProfile as u32, version as u32,
            NSOpenGLPFAColorSize as u32, (config.red_bits + config.blue_bits + config.green_bits) as u32,
            NSOpenGLPFAAlphaSize as u32, config.alpha_bits as u32,
            NSOpenGLPFADepthSize as u32, config.depth_bits as u32,
            NSOpenGLPFAStencilSize as u32, config.stencil_bits as u32,
            NSOpenGLPFAAccelerated as u32,
        ];

        if config.samples.is_some() {
            #[rustfmt::skip]
            attrs.extend_from_slice(&[
                NSOpenGLPFAMultisample as u32,
                NSOpenGLPFASampleBuffers as u32, 1,
                NSOpenGLPFASamples as u32, config.samples.unwrap() as u32,
            ]);
        }

        // **No `NSOpenGLPFADoubleBuffer`.** Nothing is swapped — the frame is
        // finished into an `IOSurface` and the surface is handed to a layer —
        // so a second buffer would only be a second copy of it.
        let _ = NSOpenGLPFADoubleBuffer;

        attrs.push(0);

        let pixel_format = NSOpenGLPixelFormat::alloc(nil).initWithAttributes_(&attrs);
        if pixel_format == nil {
            return Err(GlError::CreationFailed(()));
        }

        // A context with no view. It never presents anything itself.
        let context: id = msg_send![class!(NSOpenGLContext), alloc];
        let context: id = msg_send![context, initWithFormat: pixel_format shareContext: nil];
        let () = msg_send![pixel_format, release];
        if context == nil {
            return Err(GlError::CreationFailed(()));
        }

        context.setValues_forParameter_(
            &(config.vsync as i32),
            NSOpenGLContextParameter::NSOpenGLCPSwapInterval,
        );

        // **A layer, not a view.** A view of our own would sit over the
        // caller's and swallow the mouse; a sublayer draws in the same place
        // and takes part in no hit testing at all.
        let frame = NSView::bounds(parent_view);
        let () = msg_send![parent_view, setWantsLayer: YES];
        let parent_layer: id = msg_send![parent_view, layer];
        if parent_layer == nil {
            return Err(GlError::CreationFailed(()));
        }

        let layer: id = msg_send![class!(CALayer), layer];
        let () = msg_send![layer, retain];
        let () = msg_send![layer, setFrame: frame];
        let () = msg_send![layer, setOpaque: YES];
        // **OpenGL writes its first row at the bottom and a layer reads its
        // first row as the top**, so the picture arrives upside down unless one
        // of the two is turned over. Turning the layer over is free; flipping
        // in the renderer would be a transform on every frame.
        // No implicit animation on `contents`: a new surface every frame would
        // otherwise cross-fade with the last one.
        let () = msg_send![layer, setActions: null_actions()];
        let () = msg_send![parent_layer, addSublayer: layer];

        context.makeCurrentContext();
        let cgl: *mut c_void = msg_send![context, CGLContextObj];

        let scale = backing_scale(parent_view);
        let width = (frame.size.width * scale) as u32;
        let height = (frame.size.height * scale) as u32;

        let mut framebuffer = 0;
        glGenFramebuffers(1, &mut framebuffer);

        let surface = Surface::new(cgl, width, height)?;
        surface.attach_to(framebuffer)?;
        set_layer_contents(layer, surface.io_surface, scale);

        NSOpenGLContext::clearCurrentContext(context);

        Ok(GlContext {
            layer,
            context,
            framebuffer,
            surface: std::cell::RefCell::new(surface),
            scale: std::cell::Cell::new(scale),
        })
    }

    /// The framebuffer to render into. Its name is stable for the life of the
    /// context, so a renderer only has to be told once.
    pub fn framebuffer(&self) -> u32 {
        self.framebuffer
    }

    pub unsafe fn make_current(&self) {
        self.context.makeCurrentContext();
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer);
    }

    pub unsafe fn make_not_current(&self) {
        NSOpenGLContext::clearCurrentContext(self.context);
    }

    pub fn get_proc_address(&self, symbol: &str) -> *const c_void {
        let symbol_name = CFString::from_str(symbol).unwrap();
        let framework_name = CFString::from_str("com.apple.opengl").unwrap();
        let framework =
            unsafe { CFBundleGetBundleWithIdentifier(framework_name.as_concrete_TypeRef()) };
        let addr = unsafe {
            CFBundleGetFunctionPointerForName(framework, symbol_name.as_concrete_TypeRef())
        };
        addr as *const c_void
    }

    /// Publishes the frame.
    ///
    /// **Nothing is swapped and nothing waits.** The drawing is finished with a
    /// flush and the surface is handed to the layer, which is a CoreAnimation
    /// property assignment. There is no window-server handshake on this thread,
    /// which on a double-buffered context cost 0.6 ms typically and 13 ms at
    /// worst — on the host's run loop.
    pub fn swap_buffers(&self) {
        unsafe {
            let surface = self.surface.borrow();
            surface.publish(self.framebuffer);
            glFlush();
            set_layer_contents(self.layer, surface.io_surface, self.scale.get());
        }
    }

    /// The view and its surface follow the window.
    pub(crate) fn resize(&self, size: NSSize) {
        unsafe {
            let () = msg_send![self.layer, setFrame: NSRect::new(NSPoint::new(0.0, 0.0), size)];


            let scale = self.scale.get();
            let width = ((size.width * scale) as u32).max(1);
            let height = ((size.height * scale) as u32).max(1);
            let (have_w, have_h) = {
                let surface = self.surface.borrow();
                (surface.width, surface.height)
            };
            if have_w == width && have_h == height {
                return;
            }

            self.context.makeCurrentContext();
            let cgl: *mut c_void = msg_send![self.context, CGLContextObj];
            if let Ok(surface) = Surface::new(cgl, width, height) {
                // **The framebuffer name is kept**, only what hangs off it is
                // replaced — a renderer told about it once stays correct.
                if surface.attach_to(self.framebuffer).is_ok() {
                    let old = self.surface.replace(surface);
                    old.destroy();
                    set_layer_contents(self.layer, self.surface.borrow().io_surface, scale);
                }
            }
            NSOpenGLContext::clearCurrentContext(self.context);
        }
    }
}

unsafe fn backing_scale(view: id) -> f64 {
    let window: id = msg_send![view, window];
    if window == nil {
        1.0
    } else {
        msg_send![window, backingScaleFactor]
    }
}

unsafe fn set_layer_contents(layer: id, io_surface: *mut c_void, scale: f64) {
    let () = msg_send![layer, setContentsScale: scale];
    let () = msg_send![layer, setContents: io_surface as id];
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            self.context.makeCurrentContext();
            self.surface.borrow().destroy();
            glDeleteFramebuffers(1, &self.framebuffer);
            NSOpenGLContext::clearCurrentContext(self.context);

            let () = msg_send![self.layer, removeFromSuperlayer];
            let () = msg_send![self.context, release];
            let () = msg_send![self.layer, release];
        }
    }
}
