package org.bth;

import java.io.FileReader;
import java.nio.CharBuffer;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.lang.StringBuffer;
import java.util.ArrayList;
import android.util.Pair;
import android.util.Log;

public class SystemMonitor
{
    private static final String TAG = "harris_hessian_freak";

    public static final String[] sensors =
    {
        "/sys/devices/virtual/thermal/thermal_zone5/temp",
        "/sys/devices/virtual/thermal/thermal_zone6/temp",
        "/sys/devices/virtual/thermal/thermal_zone7/temp",
        "/sys/devices/virtual/thermal/thermal_zone8/temp",
        "/sys/devices/virtual/thermal/thermal_zone9/temp",
        "/sys/devices/virtual/thermal/thermal_zone10/temp",
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq"
    };

    public static final String[] names = 
    {
        "CPU1_TEMP",
        "CPU2_TEMP",
        "CPU3_TEMP",
        "CPU4_TEMP",
        "GPU1_TEMP",
        "GPU2_TEMP",
        "GPU_FREQ",
        "CPU0_FREQ",
        "CPU1_FREQ",
        "CPU2_FREQ",
        "CPU3_FREQ",
    };

    private ArrayList<Pair<String,FileReader>> files = new ArrayList<Pair<String,FileReader>>();

    public SystemMonitor()
    {
        for( int i = 0; i < sensors.length; i++ )
        {
            try
            {
                files.add(new Pair<String,FileReader>( names[i], new FileReader( sensors[i] )));
            }
            catch( FileNotFoundException fnfe )
            {
                Log.e( TAG,  "Unable to open " + sensors[i] );
            }
        }
    }

    public Temp[] GetTemps()
    {

        ArrayList<Temp> temps = new ArrayList<Temp>();
        for( int i = 0; i < sensors.length; i++ )
        {
            try
            {
                FileReader fr = new FileReader( sensors[i] );
                String name = names[i];
                char[] buf = new char[10];
                try
                {
                    int len = fr.read(buf);
                    String tempValue = new String( buf, 0, len-1 );
                    temps.add( new Temp( name, tempValue ) );
                    //Log.v( TAG, "temp: " + tempValue );
                }
                catch( IOException ioe )
                {
                    Log.e( TAG, "Unable to read temp file:" + name + " " + ioe.getMessage() );
                }
            }
            catch( FileNotFoundException fnfe )
            {
                Log.e( TAG,  "Unable to open " + sensors[i] );
            }
        }
        Temp[] t = {};
        return temps.toArray(t);
    }
}
