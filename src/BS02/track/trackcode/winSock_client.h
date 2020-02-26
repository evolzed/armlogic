#pragma once
#ifndef WINSOCKCLIENT
#define WINSOCKCLIENT

#include <stdio.h>

#include <iostream>
#pragma comment(lib, "ws2_32.lib")

extern void winSockclientInit();
void socksend();
void sockclose();

















#endif
