/*
* Copyright (c) 2015, Max Danielsson <work@autious.net> and Thomas Sievert
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

    public static final String[] xperia_z3_sensors =
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

    public static final String[] xperia_z3_names = 
    {
        "CPU0_TEMP",
        "CPU1_TEMP",
        "CPU2_TEMP",
        "CPU3_TEMP",
        "GPU0_TEMP",
        "GPU1_TEMP",
        "GPU_FREQ",
        "CPU0_FREQ",
        "CPU1_FREQ",
        "CPU2_FREQ",
        "CPU3_FREQ",
    };

    public static final String[] xperia_xz_sensors =
    {
        "/sys/devices/virtual/thermal/thermal_zone5/temp",
        "/sys/devices/virtual/thermal/thermal_zone6/temp",
        "/sys/devices/virtual/thermal/thermal_zone7/temp",
        "/sys/devices/virtual/thermal/thermal_zone8/temp",
        "/sys/devices/virtual/thermal/thermal_zone9/temp",
        "/sys/devices/virtual/thermal/thermal_zone4/temp",
        "/sys/devices/virtual/thermal/thermal_zone11/temp",
        "/sys/devices/virtual/thermal/thermal_zone12/temp",
        "/sys/devices/virtual/thermal/thermal_zone13/temp",
        "/sys/devices/virtual/thermal/thermal_zone14/temp",
        "/sys/devices/virtual/thermal/thermal_zone15/temp",
        "/sys/devices/virtual/thermal/thermal_zone16/temp",
        "/sys/devices/virtual/thermal/thermal_zone17/temp",
        "/sys/devices/virtual/thermal/thermal_zone18/temp",
        "/sys/devices/virtual/thermal/thermal_zone19/temp",
        "/sys/devices/virtual/thermal/thermal_zone20/temp",
        "/sys/devices/virtual/thermal/thermal_zone21/temp",
        "/sys/devices/virtual/thermal/thermal_zone22/temp",
        "/sys/devices/virtual/thermal/thermal_zone23/temp",
        "/sys/devices/virtual/thermal/thermal_zone24/temp",
        "/sys/devices/virtual/thermal/thermal_zone10/temp",
        "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq"
    };

    public static final String[] xperia_xz_names = 
    {
        "tsens_tz_sensor1",
        "tsens_tz_sensor2",
        "tsens_tz_sensor3",
        "tsens_tz_sensor4",
        "tsens_tz_sensor5",
        "tsens_tz_sensor0",
        "tsens_tz_sensor7",
        "tsens_tz_sensor8",
        "tsens_tz_sensor9",
        "tsens_tz_sensor10",
        "tsens_tz_sensor11",
        "tsens_tz_sensor12",
        "tsens_tz_sensor13",
        "tsens_tz_sensor14",
        "tsens_tz_sensor15",
        "tsens_tz_sensor16",
        "tsens_tz_sensor17",
        "tsens_tz_sensor18",
        "tsens_tz_sensor19",
        "tsens_tz_sensor20",
        "tsens_tz_sensor6",
        "GPU_FREQ",
        "CPU0_FREQ",
        "CPU1_FREQ",
        "CPU2_FREQ",
        "CPU3_FREQ",
    };

    public static final String[] sensors = xperia_xz_sensors;
    public static final String[] names = xperia_xz_names;

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
