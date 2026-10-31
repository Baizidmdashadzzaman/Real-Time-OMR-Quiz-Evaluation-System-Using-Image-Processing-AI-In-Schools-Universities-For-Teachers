package com.latticeomr.checker;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

public class MainActivity extends Activity {
    static final String CAPTURE_AUTHORITY = "com.latticeomr.checker.capture";
    private static final int REQ_PICK = 1001;
    private static final int REQ_CAMERA = 1002;

    private WebView webView;
    private ValueCallback<Uri[]> pendingCallback;
    private Uri pendingCameraUri;
    private File pendingCameraFile;
    private PermissionRequest pendingWebPermission;
    private TextToSpeech tts;
    private boolean ttsReady;
    private boolean permissionPromptActive;
    private String pendingSpeech;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.WHITE);
        getWindow().setNavigationBarColor(Color.WHITE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR);
        }

        // Temporary captures are cache files, never results. Remove leftovers
        // from an earlier session so a fresh test starts clean.
        File[] cached = getCacheDir().listFiles();
        if (cached != null) {
            for (File file : cached) {
                if (file.getName().startsWith("omr_capture_") || file.getName().startsWith("omr_pick_")) {
                    file.delete();
                }
            }
        }

        tts = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.SUCCESS || tts == null) return;
                int language = tts.setLanguage(Locale.US);
                if (language == TextToSpeech.LANG_MISSING_DATA
                        || language == TextToSpeech.LANG_NOT_SUPPORTED) {
                    tts.setLanguage(Locale.getDefault());
                }
                tts.setSpeechRate(0.98f);
                ttsReady = true;
                if (pendingSpeech != null) {
                    tts.speak(pendingSpeech, TextToSpeech.QUEUE_FLUSH, null, "omr-mark");
                    pendingSpeech = null;
                }
            }
        });

        webView = new WebView(this);
        webView.setBackgroundColor(Color.WHITE);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        // The scanner is a local file:// page.  A picked photo is copied into
        // the app cache and handed back as a file URI, which the renderer can
        // read without relying on a gallery content:// grant.
        settings.setAllowFileAccessFromFileURLs(true);
        settings.setAllowUniversalAccessFromFileURLs(true);
        settings.setTextZoom(100);
        settings.setSupportZoom(false);
        settings.setBuiltInZoomControls(false);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        settings.setSupportMultipleWindows(false);
        settings.setMediaPlaybackRequiresUserGesture(false);

        webView.addJavascriptInterface(new Bridge(), "AndroidBridge");
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        boolean wantsVideo = false;
                        for (String resource : request.getResources()) {
                            if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(resource)) {
                                wantsVideo = true;
                                break;
                            }
                        }
                        if (!wantsVideo) {
                            request.deny();
                            return;
                        }
                        if (hasCameraPermission()) {
                            request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
                        } else {
                            pendingWebPermission = request;
                            if (!permissionPromptActive) {
                                permissionPromptActive = true;
                                requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
                            }
                        }
                    }
                });
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback,
                                             FileChooserParams params) {
                if (pendingCallback != null) {
                    pendingCallback.onReceiveValue(null);
                }
                pendingCallback = callback;
                pendingCameraUri = null;
                pendingCameraFile = null;

                try {
                    if (params.isCaptureEnabled()) {
                        pendingCameraFile = File.createTempFile("omr_capture_", ".jpg", getCacheDir());
                        pendingCameraUri = Uri.parse("content://" + CAPTURE_AUTHORITY + "/"
                                + pendingCameraFile.getName());
                        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        camera.putExtra(MediaStore.EXTRA_OUTPUT, pendingCameraUri);
                        camera.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                                | Intent.FLAG_GRANT_READ_URI_PERMISSION);
                        camera.setClipData(ClipData.newRawUri("", pendingCameraUri));
                        startActivityForResult(camera, REQ_PICK);
                    } else {
                        Intent pick = new Intent(Intent.ACTION_GET_CONTENT);
                        pick.addCategory(Intent.CATEGORY_OPENABLE);
                        pick.setType("image/*");
                        startActivityForResult(Intent.createChooser(pick, "Choose a photo"), REQ_PICK);
                    }
                    return true;
                } catch (Exception error) {
                    pendingCallback = null;
                    pendingCameraUri = null;
                    pendingCameraFile = null;
                    Toast.makeText(MainActivity.this,
                            "Could not open the camera or photo picker.", Toast.LENGTH_LONG).show();
                    return false;
                }
            }
        });

        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQ_PICK) return;

        Uri result = null;
        if (resultCode == RESULT_OK) {
            if (pendingCameraUri != null && pendingCameraFile != null && pendingCameraFile.length() > 0) {
                result = Uri.fromFile(pendingCameraFile);
            } else if (data != null && data.getData() != null) {
                result = stageForWebView(data.getData());
            }
        }

        if (pendingCallback != null) {
            pendingCallback.onReceiveValue(result == null ? null : new Uri[]{result});
        }
        pendingCallback = null;
        pendingCameraUri = null;
        pendingCameraFile = null;
    }

    private Uri stageForWebView(Uri source) {
        if (source == null) return null;
        if ("file".equals(source.getScheme())) return source;
        File dest = null;
        try {
            dest = File.createTempFile("omr_pick_", ".jpg", getCacheDir());
            try (InputStream in = getContentResolver().openInputStream(source);
                 OutputStream out = new FileOutputStream(dest)) {
                if (in == null) throw new IOException("No image stream");
                byte[] buffer = new byte[32 * 1024];
                int read;
                while ((read = in.read(buffer)) > 0) out.write(buffer, 0, read);
            }
            return Uri.fromFile(dest);
        } catch (Exception error) {
            if (dest != null) dest.delete();
            Toast.makeText(this, "Could not copy the selected photo.", Toast.LENGTH_LONG).show();
            return null;
        }
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void notifyCameraPermission(boolean granted) {
        if (webView == null) return;
        String script = "window.__onCameraPermission && window.__onCameraPermission("
                + (granted ? "true" : "false") + ");";
        webView.evaluateJavascript(script, null);
    }

    private class Bridge {
        @JavascriptInterface
        public void speak(final String text) {
            if (text == null || text.isEmpty()) return;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (tts == null) return;
                    if (!ttsReady) {
                        pendingSpeech = text;
                        return;
                    }
                    tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "omr-mark");
                }
            });
        }

        @JavascriptInterface
        public void requestCamera() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (hasCameraPermission()) {
                        notifyCameraPermission(true);
                        return;
                    }
                    if (permissionPromptActive) return;
                    permissionPromptActive = true;
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
                }
            });
        }

        @JavascriptInterface
        public boolean hasCamera() {
            return hasCameraPermission();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQ_CAMERA) return;

        permissionPromptActive = false;
        boolean granted = grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED;

        if (pendingWebPermission != null) {
            if (granted) {
                pendingWebPermission.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
            } else {
                pendingWebPermission.deny();
            }
            pendingWebPermission = null;
        }
        notifyCameraPermission(granted);
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Release the camera when the app goes to the background.  A runtime
        // permission dialog can also pause the activity, so do not tear down a
        // camera that is still waiting for that answer.
        if (webView != null && !permissionPromptActive) {
            webView.evaluateJavascript(
                    "window.__onAndroidPause && window.__onAndroidPause();", null);
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            tts = null;
            ttsReady = false;
        }
        if (webView != null) {
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }
}
