package org.bth;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.TextView;
import android.widget.Button;
import android.widget.ProgressBar;
import android.content.res.AssetManager;
import android.util.Log;
import android.view.View;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.io.FileWriter;

public class HarrisHessianFreakUI extends Activity
{
    private static final String STORAGE = "/storage/sdcard0/harris_hessian_freak";
    private static final String TAG = "harris_hessian_freak";
    Button startButton;
    TextView tv;
    AssetManager mgr = null;
    HarrisHessianFreak hhf;
    ProgressBar pb;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        mgr = getResources().getAssets();
        tv = (TextView)findViewById( R.id.info_v );
        startButton = (Button)findViewById( R.id.button_id );
        pb = (ProgressBar)findViewById( R.id.progress_bar );
        hhf = new HarrisHessianFreak( mgr, tv, pb );
    }

    @Override
    public void onStart()
    {
        super.onStart();


        /*
        if( api == null )
        {
        }
        */

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

    }

    @Override
    public void onStop() {
        super.onStop();
        tv.setText( "Closed lib");
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        //api.closeLib();
        //api = null;
        AssetManager mgr = null;
    }

    public void selfDestruct( View view )
    {
        hhf.Run();


        new Thread( new Runnable()
        {  
            public void run()
            {

                try
                {
                    FileWriter fw = new FileWriter( STORAGE + "/tempdata.txt" );
                    TemperatureMonitor tm = new TemperatureMonitor();
                    long start = System.currentTimeMillis();

                    while( !hhf.IsFinished() )
                    {
                        for( Temp t : tm.GetTemps() )
                        {
                            fw.write( System.currentTimeMillis() - start + "." + t.getName() + ":" + t.getTemp() + "\n");
                        }

                        try
                        {
                            Thread.sleep(5); 
                        }
                        catch( InterruptedException ie )
                        {

                        }
                    }
                    fw.close();
                }
                catch( IOException ioe )
                {
                }
            }
        }).start();
    }
}
