import sys
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("log.txt"))  # 再添一个FileHandler

def main():
    global gState
    global gDir

    if gState == 1:
        print("gState = 1")
        time.sleep(1)
        logger.info("info")
        #logger.error("error")
        #logger.addHandler(log)
        gState = 2
    elif gState == 2:
        print("gState = 2")
        time.sleep(1)
        logger.error("error2")
        #logger.addHandler(log)
    elif gState == 3:
        print("gState = 3")
        time.sleep(1)
        logger.error("error3")

if __name__ == "__main__":
    print("please type the gState: ")
    gState = int(input())
    while (True):
        try:
            main()
        except KeyboardInterrupt:
            sys.stderr.write("Keyboard interrupt.\n")
            sys.exit(main())