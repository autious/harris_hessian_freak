package org.bth.opencltestjni;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.TextView;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class OpenCLTestJNI extends Activity
{
    private static final String TAG = "opencl_app";
    TextView tv;
    AssetManager mgr;

    public String getAlbumStorageDir() 
    {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "opencl_app_debug");
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file.getAbsolutePath();
    }    

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

        String state = Environment.getExternalStorageState();

        if( Environment.MEDIA_MOUNTED.equals(state) ) 
        {
            Log.e( TAG, "Able to save to media." );
        }
        else
        {
            Log.e( TAG, "Unable to save to media." );
        }

        if( Environment.MEDIA_MOUNTED.equals(state) || Environment.MEDIA_MOUNTED_READ_ONLY.equals(state) ) 
        {
            Log.e( TAG, "Able to read from media." );
        }
        else
        {
            Log.e( TAG, "Unable to read from media." );
        }

        Log.e( TAG, getAlbumStorageDir() );

        try
        {
            //setSaveFolder( getAlbumStorageDir().getBytes( "UTF-8" ) ); 
            String folder = new String("/storage/sdcard0/harris_hessian_freak");
            File file = new File( folder );
            file.mkdirs();
            
            setSaveFolder( folder.getBytes( "UTF-8" ) ); 
        }
        catch( UnsupportedEncodingException uee )
        {
            Log.e( TAG, uee.toString() );
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
    public native boolean setSaveFolder( byte[] path );
}
