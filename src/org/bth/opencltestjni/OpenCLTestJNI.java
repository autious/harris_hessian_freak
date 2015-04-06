package org.bth.opencltestjni;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.IOException;

public class OpenCLTestJNI extends Activity
{
    private static final String TAG = "opencl_app";
    TextView tv;
    AssetManager mgr;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        tv = new TextView(this);
        setContentView(tv);
    }

    @Override
    public void onStart()
    {
        super.onStart();

        AssetManager mgr = getResources().getAssets();

        if( initLib(mgr) )
        {
            tv.setText( "Opened lib");
            Log.v( TAG, "Opened lib" );
        }

        runTest();
    }

    @Override
    public void onStop()
    {
        super.onStop();
        tv.setText( "Closed lib");
        closeLib();
        AssetManager mgr = null;
    }

    static
    {
        System.loadLibrary("android_opencl_test_jni");
    }

    public native boolean initLib( AssetManager assetManager );
    public native boolean closeLib();
    public native String getLibError();
    public native boolean runTest();
}
