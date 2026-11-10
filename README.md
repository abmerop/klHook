# KL Hook

The purpose of this tool is to hook HIP and HSA related functions to do things when they are called.
It currently only (optionally) prints the function name if KL_HOOK_VERBOSE=1 is in the environment.

## Building

Use this command on the machine you want to use the tool on:

```sh
hipcc -std=c++17 -shared -fPIC -o libklhook.so klhook.cc -ldl
```

It defaults looking for HSA / HIP functions in /opt/rocm/.
See the environment variable if you have a non-standard ROCm install.
You can use `ldd` on a GPU application to see where the libraries libamdhip64.s0 and libhsa-runtime64.so point to.

## Running

Populate the LD_PRELOAD environment variable on the command line before your GPU application:

```sh
hipcc square.cpp -o square
LD_PRELOAD=./libklhook.so ./square
```

## Environment variables

The library uses environment variables to control a few things since libraries cannot accept command line parameters.
The following environment variables are checked:

- `export KL_HOOK_VERBOSE=1`: Enable verbose printing of the hook library to print allocations, function calls, etc.
- `export KL_HOOK_HIP_PATH="/opt/rocm/lib/libamdhip64.so"`: Specify a custom location of the HIP library to hook functions from.
- `export KL_HOOK_HSA_PATH="/opt/rocm/lib/libhsa-runtime64.so"`: Specify a custom location of the HSA library to hook functions from.

