using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Threading;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;


public class NetControl : MonoBehaviour
{
    Thread thread = null;
    string nextCommand = null; // net thread -> main thread
    byte[] lastFrame = null; // main thread -> net thread
    GameObject lion;
    GameObject head;
    GameObject jawL;
    GameObject jawR;
    Vector3 t0;
    Vector3 r0;
    Vector3 jl0;
    Vector3 jr0;
    RenderTexture rtex;
    Texture2D tex;
    void Awake()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 15;
    }
    // Start is called before the first frame update
    void Start()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 15;
        /*
        var rc = GameObject.Find("renderToTex");
        var mc = GameObject.Find("Main Camera");
        rc.transform.position = mc.transform.position;
        rc.transform.rotation = mc.transform.rotation;
        */
        var rc = GameObject.Find("Main Camera");
        var rcam = rc.GetComponent<Camera>();
        rtex = new RenderTexture(480, 480, 16, RenderTextureFormat.ARGB32);
        rcam.targetTexture =  rtex;
        tex = new Texture2D(480, 480, TextureFormat.RGB24, false);
        lion = GameObject.Find("lion");
        head = GameObject.Find("Head");
        jawL = GameObject.Find("Jaw_L.002");
        jawR = GameObject.Find("Jaw_R.002");
        t0 = lion.transform.position;
        r0 = head.transform.rotation.eulerAngles;
        jl0 = jawL.transform.rotation.eulerAngles;
        jr0 = jawL.transform.rotation.eulerAngles;
        
        if (thread == null)
        {
            thread = new Thread(RunListener);
            thread.Start();
        }
    }

    // Update is called once per frame
    void Update()
    {
        string cmd = null;
        lock(this)
        {
            cmd = nextCommand;
            nextCommand = null;
        }
        if (cmd != null)
        {
            var args = cmd.Split(' ');
            var tx = float.Parse(args[0]);
            var ty = float.Parse(args[1]);
            var tz = float.Parse(args[2]);
            var rx = float.Parse(args[3]);
            var ry = float.Parse(args[4]);
            var rz = float.Parse(args[5]);
            var jaw = float.Parse(args[6]);
            lion.transform.position = t0 + new Vector3(tx, ty, tz);
            head.transform.rotation = Quaternion.Euler(rx+r0.x, ry+r0.y, rz+r0.z);
            //jawL.transform.rotation = Quaternion.Euler(jl0.x+jaw, jl0.y, jl0.z);
            //jawR.transform.rotation = Quaternion.Euler(jr0.x+jaw, jr0.y, jr0.z);
            RenderTexture.active = rtex;
            tex.ReadPixels(new Rect(0, 0, 480, 480), 0, 0);
            RenderTexture.active = null;
            var pixels = tex.GetPixels32();
            var raw = new byte[pixels.Length * 3];
            int idx = 0;
            foreach (var p in pixels)
            {
                raw[idx++] = p.r;
                raw[idx++] = p.g;
                raw[idx++] = p.b;
            }
            lock(this)
            {
                lastFrame = raw;
            }
        }
    }
    
    void RunListener()
    {
        try
        {
            IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Parse("0.0.0.0"), 12223);
            var listener = new Socket(AddressFamily.InterNetwork,
                SocketType.Stream, ProtocolType.Tcp );
            listener.SetSocketOption(SocketOptionLevel.Socket,SocketOptionName.ReuseAddress,1);
            listener.Bind(localEndPoint);
            listener.Listen(10);
            while (true)
            {
                var handler = listener.Accept();
                while (true)
                {
                    try
                    {
                        if (!handler.Connected)
                            break;
                        var bytes = new byte[1024];
                        int bytesRec = handler.Receive(bytes);
                        if (bytesRec <= 0)
                            break;
                        var cmd = Encoding.ASCII.GetString(bytes,0,bytesRec);
                        byte[] frame = null;
                        lock(this)
                        {
                            nextCommand = cmd;
                            frame = lastFrame;
                        }
                        if (frame == null)
                            handler.Send(new byte[] { 0});
                        else
                        {
                            handler.Send(new byte[] {1});
                            handler.Send(frame);
                        }
                    }
                    catch (Exception e)
                    {
                        Debug.Log("BRONK: " + e.ToString());
                        break;
                    }
                }
            }
        }
        catch (Exception e)
        {
            Debug.Log("MEGABRONK: " + e.ToString());
        }
    }
}
