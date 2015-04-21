package org.bth;

import android.widget.TextView;
import android.widget.ProgressBar;
import android.content.res.AssetManager;
import java.lang.Thread;
import java.lang.InterruptedException;

public class HarrisHessianFreak
{
    Thread workitem;
    AssetManager mgr;
    TextView tv;
    ProgressBar pb;
    HarrisHessianFreakJNI api;

    public HarrisHessianFreak( AssetManager mgr, TextView tv, ProgressBar pb )
    {
        this.mgr = mgr;
        this.tv = tv;
        this.pb = pb;
        this.api = new HarrisHessianFreakJNI();
    }

    public void Run()
    {
        if( workitem == null )
        {
            workitem = new Thread( new Runnable() 
            {
                public void run()
                {
                    api.initLib(mgr);
                    api.runTest( new HarrisHessianProgressCallbackInterface()
                    {
                        public void progress( final int progress )
                        {
                            pb.post( new Runnable()
                            {
                                public void run()
                                {
                                    pb.setProgress( progress );
                                }
                            });
                        }
                    });

                    api.closeLib();

                    tv.post( new Runnable()
                    {
                        public void run()
                        {
                            Finish();
                        }
                    });
                }
            });

            workitem.start();
            tv.setText( "Running job." );
        }
        else
        {
            tv.setText( "Unable to run job." );
        }
    }

    public void Finish()
    {
        workitem = null;
        tv.setText( "Finished job." );
    }
}
