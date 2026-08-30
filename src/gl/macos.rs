use std::ffi::c_void;
use std::str::FromStr;

use raw_window_handle::RawWindowHandle;

use cocoa::appkit::{
    NSOpenGLContext, NSOpenGLContextParameter, NSOpenGLPFAAccelerated, NSOpenGLPFAAlphaSize,
    NSOpenGLPFABackingStore, NSOpenGLPFAColorSize, NSOpenGLPFADepthSize, NSOpenGLPFADoubleBuffer,
    NSOpenGLPFAMultisample, NSOpenGLPFAOpenGLProfile, NSOpenGLPFASampleBuffers, NSOpenGLPFASamples,
    NSOpenGLPFAStencilSize, NSOpenGLPixelFormat, NSOpenGLProfileVersion3_2Core,
    NSOpenGLProfileVersion4_1Core, NSOpenGLProfileVersionLegacy, NSOpenGLView, NSView,
};
use cocoa::base::{id, nil, YES};
use cocoa::foundation::NSSize;

use core_foundation::base::TCFType;
use core_foundation::bundle::{CFBundleGetBundleWithIdentifier, CFBundleGetFunctionPointerForName};
use core_foundation::string::CFString;

use objc::{msg_send, sel, sel_impl};

use super::{GlConfig, GlError, Profile};

pub type CreationFailedError = ();
pub struct GlContext {
    view: id,
    context: id,
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

        if config.double_buffer {
            attrs.push(NSOpenGLPFADoubleBuffer as u32);
            // **The back buffer keeps its contents across a swap.** Without
            // this, `flushBuffer` may hand back a buffer holding some earlier
            // frame, and a caller that draws only the part of the window that
            // changed gets the rest of an old one. Drawing everything every
            // frame hides that; drawing only what moved does not.
            attrs.push(NSOpenGLPFABackingStore as u32);
        }

        attrs.push(0);

        let pixel_format = NSOpenGLPixelFormat::alloc(nil).initWithAttributes_(&attrs);

        if pixel_format == nil {
            return Err(GlError::CreationFailed(()));
        }

        let view =
            NSOpenGLView::alloc(nil).initWithFrame_pixelFormat_(parent_view.frame(), pixel_format);

        if view == nil {
            return Err(GlError::CreationFailed(()));
        }

        view.setWantsBestResolutionOpenGLSurface_(YES);

        // **Layer-backed on purpose.** An `NSOpenGLView` with no layer of its
        // own, added into a window that is layer-backed — which every modern
        // host's is — makes AppKit fall back to a compatibility path for the
        // **whole window**, and the window does not come back from it: the host
        // stays sluggish until it is relaunched, long after the plugin has been
        // removed. Measured in Studio One, whose own interface draws through a
        // `CAMetalLayer`: with the editor never opened the host is fine, and
        // one open is enough to degrade it for the life of the process
        // (`docs/investigations/ui-frame-cost.md` in nxe-plugins).
        //
        // Asking for a layer explicitly keeps the surface a layer among layers
        // instead of the thing that pulls the window off its own path.
        let () = msg_send![view, setWantsLayer: YES];

        let () = msg_send![view, retain];
        NSOpenGLView::display_(view);
        parent_view.addSubview_(view);

        let context: id = msg_send![view, openGLContext];
        let () = msg_send![context, retain];

        context.setValues_forParameter_(
            &(config.vsync as i32),
            NSOpenGLContextParameter::NSOpenGLCPSwapInterval,
        );

        let () = msg_send![pixel_format, release];

        Ok(GlContext { view, context })
    }

    pub unsafe fn make_current(&self) {
        self.context.makeCurrentContext();
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

    /// Puts the frame on screen.
    ///
    /// **No `setNeedsDisplay:`.** `flushBuffer` is what presents the frame;
    /// marking the view dirty on top of it asks AppKit for a display cycle the
    /// frame does not need. For a plugin this view is a subview of the
    /// **host's** window, so that is the host's display cycle and its
    /// CoreAnimation commit, once per frame, on the thread the host also
    /// delivers input on. `resize` still marks the view, which is the case the
    /// flag was added for.
    ///
    /// **On a single-buffered context this is a `glFlush` and costs nothing.**
    /// On a double-buffered one it is a swap, and a swap hands the surface to
    /// the window server — which blocks the calling thread until the server is
    /// ready for it. Measured on an idle machine that wait was **0.6 ms
    /// typically and 13 ms at worst**, forty times a second, on whatever run
    /// loop the window was opened on. For a plugin that is the host's, and a
    /// host whose event loop is stopped in chunks that long does not track a
    /// pointer any more — it slides after it.
    pub fn swap_buffers(&self) {
        unsafe {
            self.context.flushBuffer();
        }
    }

    /// On macOS the `NSOpenGLView` needs to be resized separtely from our main view.
    pub(crate) fn resize(&self, size: NSSize) {
        unsafe { NSView::setFrameSize(self.view, size) };
        unsafe {
            let _: () = msg_send![self.view, setNeedsDisplay: YES];
        }
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            let () = msg_send![self.context, release];
            let () = msg_send![self.view, release];
        }
    }
}
