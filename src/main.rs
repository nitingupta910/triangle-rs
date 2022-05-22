use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::vk::{
    ComponentSwizzle, ImageSubresourceRange, PhysicalDevice, Queue, SurfaceKHR, SwapchainKHR,
};
use ash::{vk, Device, Entry, Instance};
use std::borrow::Cow;
use std::ffi::CStr;
use winit::window::Window;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_WIDTH: u32 = 1080;
const WINDOW_HEIGHT: u32 = 768;

struct PerFrame {}

struct Context {
    entry: Entry,
    instance: Instance,
    physical_device: PhysicalDevice,
    device: Device,
    surface_loader: Surface,
    surface: SurfaceKHR,
    swapchain_loader: Swapchain,
    swapchain: Option<SwapchainKHR>,
    swapchain_image_views: Vec<vk::ImageView>,
    per_frame: Vec<PerFrame>,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

unsafe fn init_debug(entry: &Entry, instance: &Instance) {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_call_back = debug_utils_loader
        .create_debug_utils_messenger(&debug_info, None)
        .unwrap();
}

unsafe fn init_instance(entry: &Entry, window: &Window) -> Instance {
    let app_name = CStr::from_bytes_with_nul_unchecked(b"Vulkan Triangle\0");
    let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
    let mut extension_names_raw = surface_extensions
        .iter()
        .map(|ext| *ext)
        .collect::<Vec<_>>();
    extension_names_raw.push(DebugUtils::name().as_ptr());

    let appinfo = vk::ApplicationInfo::builder()
        .application_name(app_name)
        .application_version(0)
        .engine_name(app_name)
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .enabled_layer_names(&[])
        .enabled_extension_names(&extension_names_raw);

    let instance = entry
        .create_instance(&create_info, None)
        .expect("Instance creation error");

    init_debug(entry, &instance);
    instance
}

unsafe fn init_device(
    instance: &Instance,
    surface: &SurfaceKHR,
    surface_fn: &Surface,
) -> (PhysicalDevice, Device, Queue) {
    let pdevices = instance
        .enumerate_physical_devices()
        .expect("Physical device error");
    let (pdevice, queue_family_index) = pdevices
        .iter()
        .find_map(|pdevice| {
            instance
                .get_physical_device_queue_family_properties(*pdevice)
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    let supports_graphic_and_surface = info
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && surface_fn
                            .get_physical_device_surface_support(*pdevice, index as u32, *surface)
                            .unwrap();
                    if supports_graphic_and_surface {
                        Some((*pdevice, index))
                    } else {
                        None
                    }
                })
        })
        .expect("Couldn't find suitable device.");

    let device_extension_names_raw = [Swapchain::name().as_ptr()];
    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];

    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index as u32)
        .queue_priorities(&priorities);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features);

    let device: Device = instance
        .create_device(pdevice, &device_create_info, None)
        .unwrap();

    let present_queue = device.get_device_queue(queue_family_index as u32, 0);

    (pdevice, device, present_queue)
}

fn teardown_per_frame(context: &mut Context, frame_index: usize) {}

fn init_per_frame(context: &mut Context, frame_index: usize) {}

unsafe fn init_swapchain(context: &mut Context) {
    let surface_format = context
        .surface_loader
        .get_physical_device_surface_formats(context.physical_device, context.surface)
        .unwrap()[0];

    let surface_capabilities = context
        .surface_loader
        .get_physical_device_surface_capabilities(context.physical_device, context.surface)
        .unwrap();
    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }
    let surface_resolution = match surface_capabilities.current_extent.width {
        u32::MAX => vk::Extent2D {
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
        },
        _ => surface_capabilities.current_extent,
    };
    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };
    let present_modes = context
        .surface_loader
        .get_physical_device_surface_present_modes(context.physical_device, context.surface)
        .unwrap();
    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(context.surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    let old_swapchain = context.swapchain;
    let swapchain = context
        .swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();
    context.swapchain = Some(swapchain);

    if old_swapchain.is_some() {
        let old_swapchain = old_swapchain.unwrap();
        for image_view in &context.swapchain_image_views {
            context.device.destroy_image_view(*image_view, None);
        }

        let swapchain_images = context
            .swapchain_loader
            .get_swapchain_images(old_swapchain)
            .unwrap();
        let image_count = swapchain_images.len();

        for i in 0..image_count {
            teardown_per_frame(context, i);
        }

        context.swapchain_image_views.clear();
        context
            .swapchain_loader
            .destroy_swapchain(old_swapchain, None);
    }

    let swapchain_images = context
        .swapchain_loader
        .get_swapchain_images(swapchain)
        .unwrap();
    let image_count = swapchain_images.len();
    context.per_frame.clear();
    context.per_frame.reserve(image_count);

    for i in 0..image_count {
        init_per_frame(context, i);
    }

    for i in 0..image_count {
        // Create an image view which we can render into.
        let create_info = vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .image(swapchain_images[i])
            .subresource_range(
                ImageSubresourceRange::builder()
                    .level_count(1)
                    .layer_count(1)
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .build(),
            )
            .components(
                vk::ComponentMapping::builder()
                    .r(ComponentSwizzle::R)
                    .g(ComponentSwizzle::G)
                    .b(ComponentSwizzle::B)
                    .a(ComponentSwizzle::A)
                    .build(),
            );

        let image_view = context
            .device
            .create_image_view(&create_info, None)
            .unwrap();
        context.swapchain_image_views.push(image_view);
    }
}

impl Context {
    fn new(window: &Window) -> Context {
        unsafe {
            let entry = Entry::linked();
            let instance = init_instance(&entry, &window);
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();
            let surface_loader = Surface::new(&entry, &instance);
            let (physical_device, device, queue) =
                init_device(&instance, &surface, &surface_loader);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let mut context = Context {
                entry,
                instance,
                physical_device,
                device,
                surface_loader,
                surface,
                swapchain_loader,
                swapchain: None,
                swapchain_image_views: vec![],
                per_frame: vec![],
            };

            let swapchain = init_swapchain(&mut context);
            context
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Vulkan Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build(&event_loop)
        .unwrap();

    let context = Context::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
}
