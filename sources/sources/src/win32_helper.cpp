#include "win32_helper.h"
#include <Windows.h>

int OpenFileModal( char* fname )
{
	char *filter = "All Files (*.*)\0*.*\0";
	HWND owner = NULL;
	OPENFILENAME ofn;
	char fileName[MAX_PATH];
	strcpy_s(fileName, MAX_PATH, "");
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrFile = fileName;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
	ofn.lpstrDefExt = "";
	GetOpenFileName(&ofn); 
	strcpy_s(fname, MAX_PATH, ofn.lpstrFile); 
	return strcmp(fname,"");
}

bool OpenFileModal ( std::string &out ) {
	char buffer[MAX_PATH + 1];
	int result = OpenFileModal(buffer);
	out = buffer;
	return result != 0;
}