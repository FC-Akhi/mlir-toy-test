# How to use `update-alternatives` for Install/Upgrade/Downgrade/Uninstall gcc, g++, clang, clang++, llvm-config, FileCheck

This article is for the people, who are going to get/highly involved with `C`, `C++`, `clang`, `llvm` development. Or even `python` developers or data scientists who might need to downgrade or updagrade the version of `gcc`, `g++`, `clang`, etc. for their python envrionment requirements. We will be discussing here 2 example cases (downgrade, upgrade) for respectively `gcc` + `g++` and `clang` + `llvm`.

## _How actually `update-alternatives` handles Upgrade/Downgrade of different packages (e.g. `gcc`, `clang`, `llvm-config`, etc.)_
It might sounds like "downgrading" or "upgrading", but actually you are choosing the lower/higher version of `gcc` or `g++` from a list. And `update-alternatives` is managing that list. So when you are choosing one version, other is/are still there. And it will look like following table (`gcc` example case)

```sh
  Selection    Path            Priority   Status
------------------------------------------------------------
* 0           /usr/bin/gcc-9     9       auto mode
  1           /usr/bin/gcc-7     7       manual mode
  2           /usr/bin/gcc-9     9       manual mode
```

# 1. `update-alternatives` for `gcc` and `g++`

## _1.1. Prerequisites_
- I assume you are using Ubuntu 20.04LTS. Install the following packages.
```sh
$ sudo apt-get update
$ sudo apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev
$ sudo apt-get install libgmp3-dev texinfo autoconf libtool pkg-config
```
- build-essential package will install the `g++` and `gcc` version 9.4.0 for Ubuntu 20.04. So we need to downgrade those versions to 7.5.0.


## _1.2. Downgrade gcc & g++ version from 9.4.0 to 7.5.0 using `update-alternatives`_

- **If you are in Ubuntu 18.04, then you can skip the whole step. Because 18.04 comes with 7.5.0 by default.** 
- The whole process is explained here [switch between multiple GCC and G++ compiler versions on Ubuntu 22.04](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-22-04-lts-jammy-jellyfish)
- Install `gcc-7` & `g++-7`. **Both of the versions MUST be same.**

```sh
sudo apt-get -y install gcc-7 g++-7
```

- Use `update-alternatives` tool to create list of multiple G and G++ compiler alternatives

```sh
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
```

- Check the available `gcc` compilers list. Update to `gcc-7`

```sh
$ sudo update-alternatives --config gcc

There are 2 choices for the alternative gcc (providing /usr/bin/gcc).
  Selection    Path            Priority   Status
------------------------------------------------------------
* 0           /usr/bin/gcc-9     9       auto mode
  1           /usr/bin/gcc-7     7       manual mode
  2           /usr/bin/gcc-9     9       manual mode
Press to keep the current choice[*], or type selection number:
```

- Check the available `g++` compilers list. Update to `g++-7`

```sh
$ sudo update-alternatives --config g++

There are 3 choices for the alternative g++ (providing /usr/bin/g++).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0          /usr/bin/g++-9     9       auto mode
  1          /usr/bin/g++-9     9       manual mode
  2          /usr/bin/g++-7     7       manual mode
Press to keep the current choice[*], or type selection number:
```

## _1.3. How to remove `gcc` or `g++` from update-alternatives entries and then out of OS_
**Remember, if you want packages (which were installed using `update-alternatives`) removed from your OS:**
- **Then first remove the `update-alternatives` entry.** 
- **After that, go for uninstalling them from the system using `--purge` and `apt-autoremove`.**

Please read the section **_2.4. How to remove `clang` or any package from `update-alternatives` entries and then out of OS_** for details. This follows exactly same concept.

# 2. Manage different `clang`, `clang++`, `llvm` versions (e.g. 9 & 10) using `update-alternatives`

## _2.1. Install `clang`, `clang++`, `llvm` for 9th version and make entry to `update-alternatives`_

- Install `clang-9` & `llvm-9`. **Both of the versions MUST be same.**. Actually `llvm-9` comes with `apt` package `libclang-9-dev`

```sh
sudo apt-get install clang-9 libclang-9-dev
```

- Use `update-alternatives` tool to make entries into the list for `clang-9` & `clang++-9`. **Remember, this time the `clang` we installed, is the first one in the system.**

```sh
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 9
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 9
```
- Now make entry for `llvm-config` which is the representative for whole `llvm` pack we installed

```sh
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 9
```
- And we have to also make an entry for `FileCheck`. It comes with `llvm`

```sh
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 9
```

## _2.2. Install `clang`, `clang++`, `llvm` for 10th version and make entry to `update-alternatives`_

- Install `clang-10` & `llvm-10`.

```sh
sudo apt-get install clang-10 libclang-10-dev
```

- Make entry to `update-alternatives` for all of them

```sh
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 10
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 10
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 10
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-10 10
```

## _2.3. Now let's start to play with the `update-alternatives` list with `--config` flag_

- Right now, here in the given example, they are set to `10`.

- **Remember, if you choose version `9` for `clang`, then you have to set `clang++`, `llvm-config`, `FileCheck` to `9`.**

- **For `clang`**

```sh
$ sudo update-alternatives --config clang

  Selection    Path               Priority   Status
------------------------------------------------------------
* 0            /usr/bin/clang-10   10        auto mode
  1            /usr/bin/clang-10   10        manual mode
  2            /usr/bin/clang-9    9         manual mode
Press to keep the current choice[*], or type selection number:
```

- **For `clang++`**

```sh
$ sudo update-alternatives --config clang++

  Selection    Path                 Priority   Status
------------------------------------------------------------
* 0            /usr/bin/clang++-10   10        auto mode
  1            /usr/bin/clang++-10   10        manual mode
  2            /usr/bin/clang++-9    9         manual mode
Press to keep the current choice[*], or type selection number:
```

- **For `llvm-config`**

```sh
$ sudo update-alternatives --config llvm-config

  Selection    Path                     Priority   Status
------------------------------------------------------------
* 0            /usr/bin/llvm-config-10   10        auto mode
  1            /usr/bin/llvm-config-10   10        manual mode
  2            /usr/bin/llvm-config-9    9         manual mode
Press to keep the current choice[*], or type selection number:
```

- **For `FileCheck`**

```sh
$ sudo update-alternatives --config FileCheck

  Selection    Path                   Priority   Status
------------------------------------------------------------
* 0            /usr/bin/FileCheck-10   10        auto mode
  1            /usr/bin/FileCheck-10   10        manual mode
  2            /usr/bin/FileCheck-9    9         manual mode
Press to keep the current choice[*], or type selection number:
```

## _2.4. How to remove `clang` or any package from `update-alternatives` entries and then out of OS_

**Remember, if you want packages (which were installed using `update-alternatives`) removed from your OS:**
- **Then first remove the `update-alternatives` entry.** 
- **After that, go for uninstalling them from the system using `--purge` and `auto-remove`.**

### 2.4.1. Remove one version of one package (e.g. `clang`)

```sh
sudo update-alternatives --remove clang /usr/bin/clang-9
```


### 2.4.2. Remove one package (e.g. `clang`) with all of it's version

- First check with the `--list` flag

```sh
$ update-alternatives --list clang

/usr/bin/clang-10
/usr/bin/clang-9
```
- Now remove them

```sh
sudo update-alternatives --remove-all clang
```
**[Note: BEWARE, while using `--remove-all`, will remove the pre-installed version of that software which may be a required component of your OS. Example: `python3`, which is required for some operating systems like Ubuntu using `--remove-all` with python on Ubuntu will break the OS. As you'd expect, `--remove-all`, disconnects all links to whatever package you're working with, leaving you with none in your `PATH` env variable]**

- Now run the command with `--list` or `--query` flag again

```sh
$ update-alternatives --list clang
# or
$ update-alternatives --query clang

update-alternatives: error: no alternatives for clang
```

- Now to remove the `clang-9` completely from the OS, use the following commands. It's same for `clang-10` too.

```sh
sudo apt-get purge clang-9 libclang-9-dev
sudo apt-autoremove
```

- If you want to remove entry for other packages, the process is exactly similar. 🙂
