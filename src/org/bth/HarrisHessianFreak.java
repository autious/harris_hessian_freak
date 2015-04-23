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
    HarrisHessianProgressCallback hhpc;

    boolean stop_endless_run = true;

    public HarrisHessianFreak( AssetManager mgr, TextView tv, ProgressBar pb )
    {
        this.mgr = mgr;
        this.tv = tv;
        this.pb = pb;
        this.api = new HarrisHessianFreakJNI();
        this.hhpc = new HarrisHessianProgressCallback();
        api.initLib(mgr);
    }

    public synchronized void Run( boolean endless_run )
    {
        stop_endless_run = !endless_run;
        
        if( workitem == null )
        {
            RunWorkitem( new Thread( new Runnable() 
            {
                public void run()
                {
                    api.loadImage();
                    do
                    {
                        api.runTest( hhpc );
                    } while( !IsStopped() );

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


    public synchronized void Destroy()
    {
        api.closeLib();
        api = null;
    }

    public synchronized void RunWorkitem( Thread t )
    {
        this.workitem = t;
        this.workitem.start();
    }

    private synchronized void Finish()
    {
        this.workitem = null;
        tv.setText( "Finished job." );
    }

    public synchronized boolean IsFinished()
    {
        return this.workitem == null;
    }

    private synchronized boolean IsStopped()
    {
        return stop_endless_run;
    }

    public synchronized boolean Stop()
    {
        if( stop_endless_run == false )
        {
            stop_endless_run = true;
            return true;
        }
        return false;
    }

    public class HarrisHessianProgressCallback implements HarrisHessianProgressCallbackInterface
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
    }
}
