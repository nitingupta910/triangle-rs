# triangle-rs

Vulkan triangle example using Rust (ash)

To run with Vulkan validation layer:

```shell
export VK_LAYER_PATH="$VULKAN_SDK/lib/vulkan/layers:$VK_LAYER_PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:$LD_LIBRARY_PATH"
export VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation:$VK_INSTANCE_LAYERS"

cargo run
```
