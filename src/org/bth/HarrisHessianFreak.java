package org.bth;

import android.widget.TextView;
import android.content.res.AssetManager;
import java.lang.Thread;
import java.lang.InterruptedException;

public class HarrisHessianFreak
{
    Thread workitem;
    AssetManager mgr;
    TextView tv;

    public HarrisHessianFreak( AssetManager mgr, TextView tv )
    {
        this.mgr = mgr;
        this.tv = tv;
    }

    public void Run()
    {
        if( workitem == null )
        {
            workitem = new Thread( new Runnable() 
            {
                public void run()
                {
                    try
                    {
                        Thread.sleep(2000);                    
                    }
                    catch( InterruptedException ie )
                    {

                    }

                    tv.post( new Runnable()
                    {
                        public void run()
                        {
                            HarrisHessianFreakJNI api = new HarrisHessianFreakJNI();

                            api.initLib(mgr);
                            api.runTest();
                            api.closeLib();

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
