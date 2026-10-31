package com.latticeomr.checker;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * Serves only files that already exist in this app's cache directory.  The
 * camera app receives a content:// URI with a temporary write grant, so the
 * app needs no storage permission and no captured photo is exposed elsewhere.
 */
public class CaptureProvider extends ContentProvider {
    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection,
                        String[] selectionArgs, String sortOrder) {
        return null;
    }

    @Override
    public String getType(Uri uri) {
        return "image/jpeg";
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        return null;
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection,
                      String[] selectionArgs) {
        return 0;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        String name = uri.getLastPathSegment();
        if (name == null || !name.matches("[A-Za-z0-9._-]+")) {
            throw new FileNotFoundException("Invalid capture name");
        }
        File cache = getContext().getCacheDir();
        File file = new File(cache, name);
        try {
            String base = cache.getCanonicalPath() + File.separator;
            if (!file.getCanonicalPath().startsWith(base)) {
                throw new FileNotFoundException("Capture outside cache");
            }
        } catch (IOException error) {
            throw new FileNotFoundException("Invalid capture path");
        }

        int flags = mode != null && mode.contains("w")
                ? ParcelFileDescriptor.MODE_WRITE_ONLY
                    | ParcelFileDescriptor.MODE_CREATE
                    | ParcelFileDescriptor.MODE_TRUNCATE
                : ParcelFileDescriptor.MODE_READ_ONLY;
        return ParcelFileDescriptor.open(file, flags);
    }
}
