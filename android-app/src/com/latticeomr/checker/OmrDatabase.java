package com.latticeomr.checker;

import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

/**
 * On-device SQLite database for subjects, exams and answer keys.
 *
 * Deleting a subject removes its exams, and deleting an exam removes its answer
 * key, so the teacher never leaves orphan rows behind.  Upgrades must stay
 * additive: a class list and its keys are not disposable.
 */
public class OmrDatabase extends SQLiteOpenHelper {
    private static final String NAME = "omr.db";
    private static final int VERSION = 1;

    OmrDatabase(Context context) {
        super(context, NAME, null, VERSION);
    }

    @Override
    public void onConfigure(SQLiteDatabase db) {
        super.onConfigure(db);
        db.setForeignKeyConstraintsEnabled(true);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE subjects ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "name TEXT NOT NULL UNIQUE, "
                + "created_at INTEGER NOT NULL)");

        db.execSQL("CREATE TABLE exams ("
                + "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE, "
                + "name TEXT NOT NULL, "
                + "template_id TEXT NOT NULL, "
                + "question_count INTEGER NOT NULL, "
                + "option_count INTEGER NOT NULL DEFAULT 4, "
                + "created_at INTEGER NOT NULL, "
                + "UNIQUE(subject_id, name))");

        db.execSQL("CREATE TABLE answer_keys ("
                + "exam_id INTEGER NOT NULL REFERENCES exams(id) ON DELETE CASCADE, "
                + "question_no INTEGER NOT NULL, "
                + "correct_option INTEGER NOT NULL, "
                + "marks REAL NOT NULL DEFAULT 1, "
                + "PRIMARY KEY(exam_id, question_no))");

        db.execSQL("CREATE INDEX idx_exams_subject ON exams(subject_id)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // Additive migrations only.  Version 1 is the first shipped schema, so
        // there is nothing to migrate yet; keep data instead of recreating.
    }
}
