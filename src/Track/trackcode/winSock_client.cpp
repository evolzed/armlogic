#include "stdafx.h"
#include "winSock_client.h"
#include <WinSock2.h>
#include <WS2tcpip.h>
///void winSockclientTest(SOCKET sockClient)

SOCKET sockClient;

void winSockclientInit()
{
	//加载套接字
	WSADATA wsaData;
	char buff[1024];
	memset(buff, 0, sizeof(buff));

	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		printf("Failed to load Winsock");
		return;
	}

	SOCKADDR_IN addrSrv;
	addrSrv.sin_family = AF_INET;
	//addrSrv.sin_port = htons(5099);
	addrSrv.sin_port = htons(12345);
	//addrSrv.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	long a=0;
	long *data=&a;
	//addrSrv.sin_addr.S_un.S_addr = inet_pton(AF_INET,"192.168.0.15",data);
	inet_pton(AF_INET, "192.168.0.15", data);
	addrSrv.sin_addr.S_un.S_addr = *data;

	//创建套接字
	//SOCKET sockClient = socket(AF_INET, SOCK_STREAM, 0);
	sockClient = socket(AF_INET, SOCK_STREAM, 0);
	if (SOCKET_ERROR == sockClient) {
		printf("Socket() error:%d", WSAGetLastError());
		return;
	}

	//向服务器发出连接请求
	if (connect(sockClient, (struct  sockaddr*)&addrSrv, sizeof(addrSrv)) == INVALID_SOCKET) {
		printf("Connect failed:%d", WSAGetLastError());
		return;
	}
	else
	{
		//接收数据
	//	recv(sockClient, buff, sizeof(buff), 0);
		printf("recive%s\n", buff);
	}

	//发送数据
	/*
	//char *buff1 = "hello, this is a Client....";
	char *buff1 = "1111";
	send(sockClient, buff1, sizeof(buff1), 0);

	std::cout << "send" << std::endl;
	*/
	//关闭套接字
	////closesocket(sockClient);
	////WSACleanup();
}


//发送数据
//char *buff1 = "hello, this is a Client....";
void socksend()
{
	char *buff1 = "1111";
	send(sockClient, buff1, sizeof(buff1), 0);

	std::cout << "send" << std::endl;
}



void sockclose()
{
	closesocket(sockClient);
	WSACleanup();
}