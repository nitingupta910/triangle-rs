# triangle-rs

Vulkan triangle example using Rust (ash)

To run with Vulkan validation layer:

```shell
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo run
```
(assuming you have already added `source $HOME/vulkan/latest/setup-env.sh` to your `~/.bashrc`).


To run with CLion, set the following environment variables in the 'Cargo Run' configuration. CLion does not seem to inherit from user environment.

```shell
VULKAN_SDK=/home/user/vulkan/latest/x86_64
LD_LIBRARY_PATH=/home/user/vulkan/latest/x86_64/lib
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
VK_LAYER_PATH=/home/user/vulkan/latest/x86_64/etc/vulkan/explicit_layer.d
```

Remember to adjust paths in this README to match your setup.
