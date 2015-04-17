package org.bth;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.TextView;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

public class HarrisHessianFreakTest extends Activity
{
    HarrisHessianFreakJNI api;

    private static final String TAG = "harris_hessian_freak";
    TextView tv;
    AssetManager mgr;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        api = new HarrisHessianFreakJNI();

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

        if( api.initLib(mgr) )
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

        /*
        try
        {
            //api.setSaveFolder( getAlbumStorageDir().getBytes( "UTF-8" ) ); 
            String folder = new String("/storage/sdcard0/harris_hessian_freak");
            File file = new File( folder );
            file.mkdirs();
            
            api.setSaveFolder( folder.getBytes( "UTF-8" ) ); 
        }
        catch( UnsupportedEncodingException uee )
        {
            Log.e( TAG, uee.toString() );
        }
        */

        api.runTest();
    }

    @Override
    public void onStop()
    {
        super.onStop();
        tv.setText( "Closed lib");
        api.closeLib();
        AssetManager mgr = null;
    }
}
