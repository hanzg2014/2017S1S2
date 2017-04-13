void capital(char* s)
{
    char* p = s;
    while (*p != 0){ //'\0'
    	*p -= 32;
        p++;
    }
}



