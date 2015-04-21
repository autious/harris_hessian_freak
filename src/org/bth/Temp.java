package org.bth;

public class Temp
{
    private String name;
    private int temp; 

    public Temp()
    {
        name = "";
        temp = 0;
    }

    public Temp( String name, int temp )
    {
        this.name = name;
        this.temp = temp;
    }

    public Temp( String name, String temp )
    {
        this.name = name;
        this.temp = Integer.parseInt( temp );
    }

    public int getTemp()
    {
        return temp;
    }

    public String getName()
    {
        return name;
    }
}
