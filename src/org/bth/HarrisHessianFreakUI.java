package org.bth;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.TextView;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.CheckBox;
import android.widget.Toast;
import android.content.res.AssetManager;
import android.util.Log;
import android.view.View;
import android.content.Context;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.io.FileWriter;

public class HarrisHessianFreakUI extends Activity
{
    private static final String STORAGE = "/storage/sdcard0/harris_hessian_freak";
    private static final String TAG = "harris_hessian_freak";
    TextView tv, monitoringDataTextView;
    AssetManager mgr = null;
    HarrisHessianFreak hhf;
    ProgressBar pb;
    CheckBox saveBuffersCheckbox, runIndefCheckbox;
    

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        mgr = getResources().getAssets();
        tv = (TextView)findViewById( R.id.info_v );
        monitoringDataTextView = (TextView)findViewById( R.id.monitoring_data );
        pb = (ProgressBar)findViewById( R.id.progress_bar );
        saveBuffersCheckbox = (CheckBox)findViewById( R.id.save_buffers );
        runIndefCheckbox = (CheckBox)findViewById( R.id.run_indef );
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
        AssetManager mgr = null;
        hhf.Destroy();
        hhf = null;
    }

    public void onClickStart( View view )
    {
        hhf.Run( runIndefCheckbox.isChecked() );

        new Thread( new Runnable()
        {  
            public void run()
            {
                try
                {
                    FileWriter fw = new FileWriter( STORAGE + "/monitoring_data.txt" );
                    SystemMonitor tm = new SystemMonitor();
                    long start = System.currentTimeMillis();

                    StringBuffer sb = new StringBuffer();

                    final MonitorCallbackMarker mcm = new MonitorCallbackMarker();
                    while( !hhf.IsFinished() )
                    {
                        sb.setLength(0);
                        for( Temp t : tm.GetTemps() )
                        {
                            sb.append( System.currentTimeMillis() - start + "." + t.getName() + ":" + t.getTemp() + "\n" );
                        }
                        
                        final String s = sb.toString();

                        fw.write( s );

                        if( mcm.isDone() )
                        {
                            mcm.setDone(false);
                            monitoringDataTextView.post( new Runnable()
                            {
                                public void run()
                                {
                                    monitoringDataTextView.setText( s );
                                    mcm.setDone(true);
                                }
                            });
                        }

                        try
                        {
                            Thread.sleep(50); 
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

    public void onClickStop( View view )
    {
        if( hhf.Stop() )
        {
            Context context = getApplicationContext();
            CharSequence text = "Stopping test...";
            int duration = Toast.LENGTH_SHORT;

            Toast toast = Toast.makeText(context, text, duration);
            toast.show(); 
        }
    }

    private class MonitorCallbackMarker
    {
        boolean done = true;

        public synchronized boolean isDone()
        {
            return done;
        }

        public synchronized void setDone( boolean value )
        {
            done = value; 
        }
    }
}
