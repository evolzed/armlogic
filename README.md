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

The actual source files of a software project are usually stored inside /src. 
Alternatively, you can put them into the /lib (if you're developing a library), or into the /tool (if your application's source files are not supposed to be compiled).

	main.py
		initState
			checkInitState()
				(hardware)
					powerCheck
					cameraCheck
					pressureCheck
					lightCheck
					robotCheck
				(software)
					imageCheck
					trackCheck
					TX2Check
				set global checkState
		runState
			# image.py
				loadCNN()
				testRun()
				checkImage()
			# track.py
				bgLearn()
				check()
					send img to image
					send seePic to control
				update()
					class dict
						bottleDict
							pos, type, state, t0, t1
						controlDict
							pos, type, state, t0, t1
			# control.py
				receive seePic from track.update
				get bottleDict.pos to set checkRate
				check for time
				send blast to TX2
			# TX2.py
				logPressure
				listenBlast
				pinService
			# database.py
				#mySQL dict
		debugState
			printf
			userInput
			log

## License

See [LICENSE](LICENSE) file.
 
## Package Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) file.
