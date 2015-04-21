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
            RunWorkitem( new Thread( new Runnable() 
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
            }));

            tv.setText( "Running job." );
        }
        else
        {
            tv.setText( "Unable to run job." );
        }
    }

    public synchronized void RunWorkitem( Thread t )
    {
        this.workitem = t;
        this.workitem.start();
    }

    public synchronized void Finish()
    {
        this.workitem = null;
        tv.setText( "Finished job." );
    }

    public synchronized boolean IsFinished()
    {
        return this.workitem == null;
    }
}
