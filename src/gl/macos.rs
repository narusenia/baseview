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
use cocoa::base::{id, nil, NO, YES};
use cocoa::foundation::{NSSize, NSString};

use core_foundation::base::TCFType;
use core_foundation::bundle::{CFBundleGetBundleWithIdentifier, CFBundleGetFunctionPointerForName};
use core_foundation::string::CFString;

use objc::{class, msg_send, sel, sel_impl};

use super::{GlConfig, GlError, Profile};

pub type CreationFailedError = ();

/// The GL names this backend needs. Declared here rather than pulled from a
/// loader: the OpenGL framework is linked already, and these are the handful of
/// entry points the surface path uses.
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

/// One `IOSurface`, and the framebuffer that draws into it.
struct Surface {
    io_surface: *mut c_void,
    texture: u32,
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
    /// The view whose layer shows the surface. Owned by the caller's view
    /// hierarchy; this holds a retain of its own.
    view: id,
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

        let mut texture = 0;
        glGenTextures(1, &mut texture);
        glBindTexture(GL_TEXTURE_RECTANGLE, texture);
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

        // femtovg's `set_screen_target` requires depth and stencil, and vizia
        // clips with the stencil buffer.
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

        Ok(Surface { io_surface, texture, depth_stencil, width, height })
    }

    unsafe fn attach_to(&self, framebuffer: u32) -> Result<(), GlError> {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_RECTANGLE,
            self.texture,
            0,
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

    unsafe fn destroy(&self) {
        glDeleteTextures(1, &self.texture);
        glDeleteRenderbuffers(1, &self.depth_stencil);
        let () = msg_send![self.io_surface as id, release];
    }
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

        // The view that shows the surface: layer-backed, and nothing else.
        let frame = NSView::frame(parent_view);
        let view: id = msg_send![class!(NSView), alloc];
        let view: id = msg_send![view, initWithFrame: frame];
        if view == nil {
            return Err(GlError::CreationFailed(()));
        }
        let () = msg_send![view, setWantsLayer: YES];
        // The surface is drawn with its origin at the bottom, the way OpenGL
        // leaves it; the layer is told to read it that way rather than the
        // picture being flipped on the way in.
        let () = msg_send![view, setLayerContentsPlacement: 0i64];
        let layer: id = msg_send![view, layer];
        let () = msg_send![layer, setContentsGravity: NSString::alloc(nil).init_str("bottomLeft")];
        let () = msg_send![layer, setOpaque: YES];

        let () = msg_send![view, retain];
        parent_view.addSubview_(view);

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
            view,
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
            glFlush();
            let layer: id = msg_send![self.view, layer];
            set_layer_contents(layer, self.surface.borrow().io_surface, self.scale.get());
        }
    }

    /// The view and its surface follow the window.
    pub(crate) fn resize(&self, size: NSSize) {
        unsafe {
            NSView::setFrameSize(self.view, size);

            let parent: id = msg_send![self.view, superview];
            let scale = if parent == nil { self.scale.get() } else { backing_scale(parent) };
            self.scale.set(scale);

            let width = ((size.width * scale) as u32).max(1);
            let height = ((size.height * scale) as u32).max(1);
            if self.surface.borrow().width == width && self.surface.borrow().height == height {
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
                    let layer: id = msg_send![self.view, layer];
                    set_layer_contents(layer, self.surface.borrow().io_surface, scale);
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

            let () = msg_send![self.view, removeFromSuperview];
            let () = msg_send![self.context, release];
            let () = msg_send![self.view, release];
        }
    }
}
