package org.bth;

import android.content.res.AssetManager;

public class HarrisHessianFreakJNI
{
    static
    {
        System.loadLibrary("harris_hessian_freak_jni");
    }

    public native boolean initLib( AssetManager assetManager );
    public native boolean closeLib();
    public native String getLibError();
    public native boolean loadImage();
    public native boolean runTest( HarrisHessianProgressCallbackInterface hhpci );
    public native boolean setSaveFolder( byte[] path );
}
