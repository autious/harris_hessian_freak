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

    public synchronized void Run( boolean endless_run, boolean workgroup_test_run )
    {
        stop_endless_run = !endless_run;
        
        if( workitem == null )
        {
            RunWorkitem( new Thread( new Runnable() 
            {
                public void run()
                {
                    static int[] workgroup_list = {
                        2,2,
                        4,4,
                        8,8,
                        1,32,
                        2,16,
                        4,8,
                        8,4,
                        16,2,
                        32,1};

                    int size = workgroup_list.length/2;
                    int count = 0;
                    boolean repeat_run = false;
                    int rep_count = 0;
                    int rep_limit = 10;

                    api.loadImage();
                    do
                    {
                        if( workgroup_test_run )
                        {
                            api.setGaussXWorkgroup( workgroup_list[count*2], workgroup_list[count*2+1] );
                            api.setGaussYWorkgroup( workgroup_list[count*2], workgroup_list[count*2+1] );
                            
                            if( rep_count >= rep_limit )
                            {
                                count++;
                                rep_count = 0;
                            }
                            else
                            {
                                rep_count++;
                            }

                            if( count < size )
                            {
                                repeat_run = true;
                            }
                            else
                            {
                                repeat_run = false;
                            }
                        }
                        api.runTest( hhpc );
                    } while( !IsStopped() && repeat_run );

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
