use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::prelude::VkResult;
use ash::vk::{
    ComponentSwizzle, Format, ImageSubresourceRange, PhysicalDevice, Pipeline, PipelineLayout,
    Queue, RenderPass, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
};
use ash::{vk, Device, Entry, Instance};
use std::borrow::Cow;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use winit::window::Window;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_WIDTH: u32 = 1080;
const WINDOW_HEIGHT: u32 = 768;

struct Context {
    entry: Entry,

    /// The Vulkan instance.
    instance: Instance,

    /// The Vulkan physical device.
    physical_device: PhysicalDevice,

    /// The Vulkan device.
    device: Device,

    queue: Queue,

    /// The queue family index where graphics work will be submitted.
    graphics_queue_index: u32,

    surface_loader: Surface,

    /// The surface we will render to.
    surface: SurfaceKHR,

    surface_resolution: vk::Extent2D,

    surface_format: Option<SurfaceFormatKHR>,

    swapchain_loader: Swapchain,

    /// The swapchain.
    swapchain: SwapchainKHR,

    /// The image view for each swapchain image.
    swapchain_image_views: Vec<vk::ImageView>,

    /// The framebuffer for each swapchain image view.
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    /// The renderpass description.
    render_pass: RenderPass,

    /// The graphics pipeline.
    pipeline: Pipeline,

    /**
     * The pipeline layout for resources.
     * Not used in this sample, but we still need to provide a dummy one.
     */
    pipeline_layout: PipelineLayout,

    /// A set of semaphores that can be reused.
    recycled_semaphores: Vec<vk::Semaphore>,

    /// A set of per-frame data.
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
) -> (PhysicalDevice, Device, Queue, u32) {
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

    (pdevice, device, present_queue, queue_family_index as u32)
}

struct PerFrame {
    queue_submit_fence: vk::Fence,
    primary_command_pool: vk::CommandPool,
    primary_command_buffer: vk::CommandBuffer,
    swapchain_acquire_semaphore: vk::Semaphore,
    swapchain_release_semaphore: vk::Semaphore,
}

impl PerFrame {
    unsafe fn new(context: &Context) -> PerFrame {
        let info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let queue_submit_fence = context.device.create_fence(&info, None).unwrap();

        let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(context.graphics_queue_index);
        let primary_command_pool = context
            .device
            .create_command_pool(&cmd_pool_info, None)
            .unwrap();

        let cmd_buf_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(primary_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let primary_command_buffer = context
            .device
            .allocate_command_buffers(&cmd_buf_info)
            .unwrap()[0];

        let swapchain_acquire_semaphore = vk::Semaphore::null();
        let swapchain_release_semaphore = vk::Semaphore::null();

        PerFrame {
            queue_submit_fence,
            primary_command_pool,
            primary_command_buffer,
            swapchain_acquire_semaphore,
            swapchain_release_semaphore,
        }
    }
}

unsafe fn teardown_per_frame(context: &mut Context, frame_index: usize) {
    let mut per_frame = &mut context.per_frame[frame_index];

    context
        .device
        .destroy_fence(per_frame.queue_submit_fence, None);
    per_frame.queue_submit_fence = vk::Fence::null();

    context.device.free_command_buffers(
        per_frame.primary_command_pool,
        &[per_frame.primary_command_buffer],
    );
    per_frame.primary_command_buffer = vk::CommandBuffer::null();

    if per_frame.swapchain_acquire_semaphore != vk::Semaphore::null() {
        context
            .device
            .destroy_semaphore(per_frame.swapchain_acquire_semaphore, None);
        per_frame.swapchain_acquire_semaphore = vk::Semaphore::null();
    }

    if per_frame.swapchain_release_semaphore != vk::Semaphore::null() {
        context
            .device
            .destroy_semaphore(per_frame.swapchain_release_semaphore, None);
        per_frame.swapchain_release_semaphore = vk::Semaphore::null();
    }
}

unsafe fn init_swapchain(context: &mut Context) {
    let surface_format = context
        .surface_loader
        .get_physical_device_surface_formats(context.physical_device, context.surface)
        .unwrap()[0];
    context.surface_format = Some(surface_format);

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
    context.surface_resolution = surface_resolution;

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

    context.swapchain = context
        .swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();

    if old_swapchain != SwapchainKHR::null() {
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
        .get_swapchain_images(context.swapchain)
        .unwrap();
    let image_count = swapchain_images.len();
    context.per_frame.clear();
    context.per_frame.reserve(image_count);

    for i in 0..image_count {
        context.per_frame.push(PerFrame::new(context));
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

unsafe fn init_render_pass(context: &mut Context) {
    let surface_format = context.surface_format.unwrap();
    let attachments = [vk::AttachmentDescription::builder()
        // Backbuffer format.
        .format(surface_format.format)
        // Not multisampled.
        .samples(vk::SampleCountFlags::TYPE_1)
        // When starting the frame, we want tiles to be cleared.
        .load_op(vk::AttachmentLoadOp::CLEAR)
        // When ending the frame, we want tiles to be written out.
        .store_op(vk::AttachmentStoreOp::STORE)
        // Don't care about stencil since we're not using it.
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // The image layout will be undefined when the render pass begins.
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // After the render pass is complete, we will transition to PRESENT_SRC_KHR layout.
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    // We have one subpass. This subpass has one color attachment.
    // While executing this subpass, the attachment will be in attachment optimal layout.
    let color_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    // We will end up with two transitions.
    // The first one happens right before we start subpass #0, where
    // UNDEFINED is transitioned into COLOR_ATTACHMENT_OPTIMAL.
    // The final layout in the render pass attachment states PRESENT_SRC_KHR, so we
    // will get a final transition from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR.
    let subpasses = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .build()];

    // Create a dependency to external events.
    // We need to wait for the WSI semaphore to signal.
    // Only pipeline stages which depend on COLOR_ATTACHMENT_OUTPUT_BIT will
    // actually wait for the semaphore, so we must also wait for that pipeline stage.
    let dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        // Since we changed the image layout, we need to make the memory visible to
        // color attachment to modify.
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let rp_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    context.render_pass = context.device.create_render_pass(&rp_info, None).unwrap();
}

/**
 * @brief Initializes the Vulkan pipeline.
 * @param context A Vulkan context with a device and a render pass already set up.
 */
unsafe fn init_pipeline(context: &mut Context) {
    // Create a blank pipeline layout.
    // We are not binding any resources to the pipeline in this first sample.
    let layout_info = vk::PipelineLayoutCreateInfo::default();
    context.pipeline_layout = context
        .device
        .create_pipeline_layout(&layout_info, None)
        .unwrap();

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    // Specify we will use triangle lists to draw geometry.
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    // Specify rasterization state.
    let raster = vk::PipelineRasterizationStateCreateInfo::builder()
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);

    // Our attachment will write to all color channels, but no blending is enabled.
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .build()];

    let blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&blend_attachments)
        .build();

    let viewport = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1)
        .build();

    // Disable all depth testing.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

    // No multisampling.
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let dynamic = vk::PipelineDynamicStateCreateInfo::builder()
        // Specify that these states will be dynamic, i.e. not part of pipeline state object.
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    // Load our SPIR-V shaders.
    let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(
        vk_shader_macros::include_glsl!("shader/triangle.vert", kind: vert),
    );

    let frag_shader_info = vk::ShaderModuleCreateInfo::builder()
        .code(vk_shader_macros::include_glsl!("shader/triangle.frag"));

    let vertex_shader_module = context
        .device
        .create_shader_module(&vertex_shader_info, None)
        .expect("Vertex shader module error");

    let fragment_shader_module = context
        .device
        .create_shader_module(&frag_shader_info, None)
        .expect("Fragment shader module error");

    let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
    let shader_stage_create_infos = [
        vk::PipelineShaderStageCreateInfo {
            module: vertex_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            module: fragment_shader_module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let pipe = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .rasterization_state(&raster)
        .color_blend_state(&blend)
        .multisample_state(&multisample)
        .viewport_state(&viewport)
        .depth_stencil_state(&depth_stencil)
        .dynamic_state(&dynamic)
        // We need to specify the pipeline layout and the render pass description up front as well.
        .render_pass(context.render_pass)
        .layout(context.pipeline_layout)
        .build()];

    context.pipeline = context
        .device
        .create_graphics_pipelines(vk::PipelineCache::default(), &pipe, None)
        .unwrap()[0];

    context
        .device
        .destroy_shader_module(vertex_shader_module, None);

    context
        .device
        .destroy_shader_module(fragment_shader_module, None);
}

unsafe fn init_framebuffers(context: &mut Context) {
    // Create framebuffer for each swapchain image view
    for image_view in &context.swapchain_image_views {
        let attachments = [*image_view];
        let fb_info = vk::FramebufferCreateInfo::builder()
            .render_pass(context.render_pass)
            .attachments(&attachments)
            .width(context.surface_resolution.width)
            .height(context.surface_resolution.height)
            .layers(1);

        let fb = context.device.create_framebuffer(&fb_info, None).unwrap();
        context.swapchain_framebuffers.push(fb);
    }
}

impl Context {
    fn new(window: &Window) -> Context {
        unsafe {
            let entry = Entry::linked();
            let instance = init_instance(&entry, &window);
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();
            let surface_loader = Surface::new(&entry, &instance);
            let (physical_device, device, queue, graphics_queue_index) =
                init_device(&instance, &surface, &surface_loader);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let mut context = Context {
                entry,
                instance,
                physical_device,
                device,
                queue,
                graphics_queue_index,
                surface_loader,
                surface,
                surface_resolution: vk::Extent2D::default(),
                swapchain_loader,
                swapchain: SwapchainKHR::null(),
                swapchain_image_views: vec![],
                surface_format: None,
                per_frame: vec![],
                render_pass: RenderPass::null(),
                pipeline: Pipeline::null(),
                pipeline_layout: PipelineLayout::null(),
                swapchain_framebuffers: vec![],
                recycled_semaphores: vec![],
            };

            init_swapchain(&mut context);
            init_render_pass(&mut context);
            init_pipeline(&mut context);
            init_framebuffers(&mut context);
            context
        }
    }
}

unsafe fn acquire_next_image(context: &mut Context) -> VkResult<u32> {
    let acquire_semaphore = match context.recycled_semaphores.pop() {
        Some(semaphore) => semaphore,
        None => {
            let info = vk::SemaphoreCreateInfo::default();
            context.device.create_semaphore(&info, None).unwrap()
        }
    };

    let image_index = match context.swapchain_loader.acquire_next_image(
        context.swapchain,
        u64::MAX,
        acquire_semaphore,
        vk::Fence::null(),
    ) {
        Ok((image_index, _)) => image_index,
        Err(e) => {
            context.recycled_semaphores.push(acquire_semaphore);
            return Err(e);
        }
    };

    // If we have outstanding fences for this swapchain image, wait for them to complete first.
    // After begin frame returns, it is safe to reuse or delete resources which
    // were used previously.
    //
    // We wait for fences which completes N frames earlier, so we do not stall,
    // waiting for all GPU work to complete before this returns.
    // Normally, this doesn't really block at all,
    // since we're waiting for old frames to have been completed, but just in case.
    let per_frame = &mut context.per_frame[image_index as usize];
    let fences = [per_frame.queue_submit_fence];
    context
        .device
        .wait_for_fences(&fences, true, u64::MAX)
        .unwrap();
    context.device.reset_fences(&fences).unwrap();

    context
        .device
        .reset_command_pool(
            per_frame.primary_command_pool,
            vk::CommandPoolResetFlags::empty(),
        )
        .unwrap();

    // Recycle the old semaphore back into the semaphore manager.
    let old_semaphore = per_frame.swapchain_acquire_semaphore;
    if old_semaphore != vk::Semaphore::null() {
        context.recycled_semaphores.push(old_semaphore);
    }

    per_frame.swapchain_acquire_semaphore = acquire_semaphore;

    Ok(image_index)
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
