# BottleSort version 0.1

## Description

This is the armlogic "src/BS0.1"-feed containing community-maintained build scripts, options and patches for applications, modules and libraries used within armlogic.

## Usage

This repository is intended to be layered on-top of an armlogic buildroot. If you do not have an armlogic buildroot installed, see the documentation at: [armlogic Buildroot – Installation](https://armlogic.tech/Buildroot) on the armlogic support site.

This feed is enabled by default. To install all its package definitions, run:
```
src/BS0.1 update packages
src/BS0.1 install -a -p packages
```

## Source files

![srcTODO](https://github.com/evolzed/armlogic/blob/BottleSort0.1/src/srcTODO.txt)
The actual source files of a software project are usually stored inside /src. 
Alternatively, you can put them into the /lib (if you're developing a library), or into the /tool (if your application's source files are not supposed to be compiled).

## License

See [LICENSE](LICENSE) file.
 
## Package Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) file.
