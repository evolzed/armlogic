# armlogic

## Description

This is the armlogic "packages"-feed containing community-maintained build scripts, options and patches for applications, modules and libraries used within armligic.

## Usage

This repository is intended to be layered on-top of an armlogic buildroot. If you do not have an armlogic buildroot installed, see the documentation at: [armlogic Buildroot – Installation](https://openwrt.org/docs/guide-developer/build-system/install-buildsystem) on the armlogicsupport site.

This feed is enabled by default. To install all its package definitions, run:
```
./scripts/feeds update packages
./scripts/feeds install -a -p packages
```

## License

See [LICENSE](LICENSE) file.
 
## Package Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) file.
