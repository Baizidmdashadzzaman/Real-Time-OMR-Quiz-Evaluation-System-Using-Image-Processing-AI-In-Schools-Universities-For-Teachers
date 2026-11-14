package com.latticeomr.checker;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteConstraintException;
import android.database.sqlite.SQLiteDatabase;
import android.webkit.JavascriptInterface;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * JSON bridge between the WebView screens and the on-device SQLite database.
 *
 * Every call returns a JSON string (or {"error": "..."}), so the JavaScript
 * side needs no native SDK and the app stays fully offline.
 */
public class DatabaseBridge {
    private final OmrDatabase helper;

    DatabaseBridge(Context context) {
        this.helper = new OmrDatabase(context.getApplicationContext());
    }

    private SQLiteDatabase db() {
        return helper.getWritableDatabase();
    }

    private static String error(String message) {
        JSONObject json = new JSONObject();
        try {
            json.put("error", message == null ? "The database rejected the request." : message);
        } catch (JSONException ignored) {
            // A plain message is still better than a crash.
        }
        return json.toString();
    }

    private static int count(Cursor cursor) {
        return cursor == null ? 0 : cursor.getCount();
    }

    /* ------------------------------------------------------------- subjects */

    @JavascriptInterface
    public String listSubjects() {
        Cursor cursor = null;
        try {
            JSONArray out = new JSONArray();
            cursor = db().rawQuery(
                    "SELECT s.id, s.name, "
                            + "(SELECT COUNT(*) FROM exams e WHERE e.subject_id = s.id) "
                            + "FROM subjects s ORDER BY s.name COLLATE NOCASE", null);
            while (cursor.moveToNext()) {
                JSONObject row = new JSONObject();
                row.put("id", cursor.getLong(0));
                row.put("name", cursor.getString(1));
                row.put("examCount", cursor.getInt(2));
                out.put(row);
            }
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    @JavascriptInterface
    public String addSubject(String rawName) {
        String name = rawName == null ? "" : rawName.trim();
        if (name.isEmpty()) return error("Enter a subject name.");
        try {
            ContentValues values = new ContentValues();
            values.put("name", name);
            values.put("created_at", System.currentTimeMillis());
            long id = db().insertOrThrow("subjects", null, values);
            JSONObject out = new JSONObject();
            out.put("id", id);
            return out.toString();
        } catch (SQLiteConstraintException e) {
            return error("A subject with that name already exists.");
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    @JavascriptInterface
    public String renameSubject(int id, String rawName) {
        String name = rawName == null ? "" : rawName.trim();
        if (name.isEmpty()) return error("Enter a subject name.");
        try {
            ContentValues values = new ContentValues();
            values.put("name", name);
            int changed = db().update("subjects", values, "id = ?",
                    new String[]{String.valueOf(id)});
            if (changed == 0) return error("That subject no longer exists.");
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (SQLiteConstraintException e) {
            return error("A subject with that name already exists.");
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    @JavascriptInterface
    public String deleteSubject(int id) {
        try {
            db().delete("subjects", "id = ?", new String[]{String.valueOf(id)});
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    /* ---------------------------------------------------------------- exams */

    @JavascriptInterface
    public String listExams(int subjectId) {
        Cursor cursor = null;
        try {
            JSONArray out = new JSONArray();
            cursor = db().rawQuery(
                    "SELECT e.id, e.subject_id, e.name, e.template_id, e.question_count, "
                            + "e.option_count, "
                            + "(SELECT COUNT(*) FROM answer_keys k WHERE k.exam_id = e.id) "
                            + "FROM exams e WHERE e.subject_id = ? "
                            + "ORDER BY e.created_at DESC",
                    new String[]{String.valueOf(subjectId)});
            while (cursor.moveToNext()) {
                JSONObject row = new JSONObject();
                row.put("id", cursor.getLong(0));
                row.put("subjectId", cursor.getLong(1));
                row.put("name", cursor.getString(2));
                row.put("templateId", cursor.getString(3));
                row.put("questionCount", cursor.getInt(4));
                row.put("optionCount", cursor.getInt(5));
                row.put("keyCount", cursor.getInt(6));
                out.put(row);
            }
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        } finally {
            if (cursor != null) cursor.close();
        }
    }

    @JavascriptInterface
    public String addExam(int subjectId, String rawName, String rawCount, String templateId) {
        String name = rawName == null ? "" : rawName.trim();
        if (name.isEmpty()) return error("Enter a name for the exam.");
        int questionCount;
        try {
            questionCount = Integer.parseInt(rawCount == null ? "" : rawCount.trim());
        } catch (NumberFormatException e) {
            return error("Pick a question count for the exam.");
        }
        if (questionCount < 1 || questionCount > 200) {
            return error("Question count must be between 1 and 200.");
        }
        String template = templateId == null || templateId.trim().isEmpty()
                ? "omr_" + questionCount
                : templateId.trim();
        try {
            Cursor subject = db().rawQuery("SELECT id FROM subjects WHERE id = ?",
                    new String[]{String.valueOf(subjectId)});
            boolean exists = count(subject) > 0;
            if (subject != null) subject.close();
            if (!exists) return error("Pick a subject first.");

            ContentValues values = new ContentValues();
            values.put("subject_id", subjectId);
            values.put("name", name);
            values.put("template_id", template);
            values.put("question_count", questionCount);
            values.put("option_count", 4);
            values.put("created_at", System.currentTimeMillis());
            long id = db().insertOrThrow("exams", null, values);
            JSONObject out = new JSONObject();
            out.put("id", id);
            return out.toString();
        } catch (SQLiteConstraintException e) {
            return error("This subject already has an exam with that name.");
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    @JavascriptInterface
    public String updateExam(int id, String rawName, String rawCount, String templateId) {
        try {
            ContentValues values = new ContentValues();
            if (rawName != null) {
                String name = rawName.trim();
                if (name.isEmpty()) return error("Enter a name for the exam.");
                values.put("name", name);
            }
            if (rawCount != null && !rawCount.trim().isEmpty()) {
                int count = Integer.parseInt(rawCount.trim());
                if (count < 1 || count > 200) {
                    return error("Question count must be between 1 and 200.");
                }
                values.put("question_count", count);
                values.put("template_id", templateId == null || templateId.trim().isEmpty()
                        ? "omr_" + count
                        : templateId.trim());
            }
            if (values.size() == 0) {
                JSONObject out = new JSONObject();
                out.put("ok", true);
                return out.toString();
            }
            int changed = db().update("exams", values, "id = ?",
                    new String[]{String.valueOf(id)});
            if (changed == 0) return error("That exam no longer exists.");
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (SQLiteConstraintException e) {
            return error("This subject already has an exam with that name.");
        } catch (NumberFormatException e) {
            return error("That question count is not a number.");
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    @JavascriptInterface
    public String deleteExam(int id) {
        try {
            db().delete("exams", "id = ?", new String[]{String.valueOf(id)});
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }

    /* ----------------------------------------------------------- answer keys */

    @JavascriptInterface
    public String saveKey(int examId, String answersJson, String marksJson) {
        SQLiteDatabase database = db();
        database.beginTransaction();
        try {
            JSONArray answers = new JSONArray(answersJson == null ? "[]" : answersJson);
            JSONArray marks = new JSONArray(marksJson == null ? "[]" : marksJson);

            database.delete("answer_keys", "exam_id = ?",
                    new String[]{String.valueOf(examId)});

            for (int i = 0; i < answers.length(); i++) {
                if (answers.isNull(i)) continue;
                int option = answers.getInt(i);
                if (option < 0) continue;
                double mark = 1.0;
                if (i < marks.length() && !marks.isNull(i)) {
                    mark = marks.getDouble(i);
                    if (mark < 0) mark = 0;
                }
                ContentValues values = new ContentValues();
                values.put("exam_id", examId);
                values.put("question_no", i + 1);
                values.put("correct_option", option);
                values.put("marks", mark);
                database.insertOrThrow("answer_keys", null, values);
            }
            database.setTransactionSuccessful();
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        } finally {
            database.endTransaction();
        }
    }

    @JavascriptInterface
    public String loadKey(int examId) {
        Cursor exam = null;
        Cursor keys = null;
        try {
            int questionCount = 0;
            exam = db().rawQuery("SELECT question_count FROM exams WHERE id = ?",
                    new String[]{String.valueOf(examId)});
            if (exam.moveToFirst()) questionCount = exam.getInt(0);

            JSONArray answers = new JSONArray();
            JSONArray marks = new JSONArray();
            for (int i = 0; i < questionCount; i++) {
                answers.put(JSONObject.NULL);
                marks.put(1.0);
            }

            keys = db().rawQuery(
                    "SELECT question_no, correct_option, marks FROM answer_keys "
                            + "WHERE exam_id = ? ORDER BY question_no",
                    new String[]{String.valueOf(examId)});
            while (keys.moveToNext()) {
                int index = keys.getInt(0) - 1;
                if (index < 0 || index >= questionCount) continue;
                answers.put(index, keys.getInt(1));
                marks.put(index, keys.getDouble(2));
            }

            JSONObject out = new JSONObject();
            out.put("answers", answers);
            out.put("marks", marks);
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        } finally {
            if (exam != null) exam.close();
            if (keys != null) keys.close();
        }
    }

    @JavascriptInterface
    public String clearKey(int examId) {
        try {
            db().delete("answer_keys", "exam_id = ?", new String[]{String.valueOf(examId)});
            JSONObject out = new JSONObject();
            out.put("ok", true);
            return out.toString();
        } catch (Exception e) {
            return error(e.getMessage());
        }
    }
}
