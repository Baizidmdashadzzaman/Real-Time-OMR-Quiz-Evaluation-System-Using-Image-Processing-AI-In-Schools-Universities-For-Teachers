/*
 * Local storage for subjects, exams, answer keys and per-question marks.
 *
 * On the phone everything goes through the Android bridge into SQLite
 * (DatabaseBridge -> OmrDatabase).  Inside a plain browser — used for
 * development — the same API is emulated with localStorage so the screens can
 * be exercised without a device.
 */
(function (root) {
  'use strict';

  const hasBridge = () => Boolean(
    root.OmrDatabase && typeof root.OmrDatabase.listSubjects === 'function'
  );

  function call(method) {
    const args = Array.prototype.slice.call(arguments, 1);
    const bridge = root.OmrDatabase;
    let raw;
    try {
      raw = bridge[method].apply(bridge, args);
    } catch (error) {
      throw new Error((error && error.message) || 'The phone database refused the request.');
    }
    let parsed = null;
    try {
      parsed = JSON.parse(raw === undefined || raw === null || raw === '' ? 'null' : raw);
    } catch (error) {
      throw new Error('The phone database returned an unreadable answer.');
    }
    if (parsed && parsed.error) throw new Error(parsed.error);
    return parsed;
  }

  /* ------------------------------------------------------- browser fallback */

  const FALLBACK_KEY = 'omr.local.db.v1';

  function fresh() {
    return { nextSubject: 1, nextExam: 1, subjects: [], exams: [], keys: {} };
  }

  // Keeps the browser preview usable even where localStorage is blocked
  // (private mode, file:// in some browsers).
  let memoryDb = null;

  function readFallback() {
    if (memoryDb) return memoryDb;
    try {
      const raw = root.localStorage && root.localStorage.getItem(FALLBACK_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      memoryDb = parsed && parsed.subjects ? parsed : fresh();
      return memoryDb;
    } catch (error) {
      memoryDb = fresh();
      return memoryDb;
    }
  }

  function writeFallback(db) {
    memoryDb = db;
    try {
      if (root.localStorage) root.localStorage.setItem(FALLBACK_KEY, JSON.stringify(db));
    } catch (error) {
      // Private-mode browsers simply do not persist; the screens still work.
    }
  }

  const now = () => Date.now();

  function fallback(method, args) {
    const db = readFallback();
    switch (method) {
      case 'listSubjects':
        return db.subjects
          .slice()
          .sort((a, b) => a.name.localeCompare(b.name))
          .map(s => ({
            id: s.id,
            name: s.name,
            examCount: db.exams.filter(e => e.subjectId === s.id).length,
          }));

      case 'addSubject': {
        const name = String(args[0] || '').trim();
        if (!name) throw new Error('Enter a subject name.');
        if (db.subjects.some(s => s.name.toLowerCase() === name.toLowerCase())) {
          throw new Error('A subject with that name already exists.');
        }
        const subject = { id: db.nextSubject++, name, createdAt: now() };
        db.subjects.push(subject);
        writeFallback(db);
        return { id: subject.id };
      }

      case 'renameSubject': {
        const id = Number(args[0]);
        const name = String(args[1] || '').trim();
        if (!name) throw new Error('Enter a subject name.');
        const subject = db.subjects.find(s => s.id === id);
        if (!subject) throw new Error('That subject no longer exists.');
        if (db.subjects.some(s => s.id !== id && s.name.toLowerCase() === name.toLowerCase())) {
          throw new Error('A subject with that name already exists.');
        }
        subject.name = name;
        writeFallback(db);
        return { ok: true };
      }

      case 'deleteSubject': {
        const id = Number(args[0]);
        db.subjects = db.subjects.filter(s => s.id !== id);
        const gone = db.exams.filter(e => e.subjectId === id).map(e => e.id);
        db.exams = db.exams.filter(e => e.subjectId !== id);
        gone.forEach(examId => { delete db.keys[examId]; });
        writeFallback(db);
        return { ok: true };
      }

      case 'listExams': {
        const subjectId = Number(args[0]);
        return db.exams
          .filter(e => e.subjectId === subjectId)
          .sort((a, b) => b.createdAt - a.createdAt)
          .map(e => ({
            id: e.id,
            subjectId: e.subjectId,
            name: e.name,
            templateId: e.templateId,
            questionCount: e.questionCount,
            optionCount: e.optionCount,
            keyCount: Object.keys(db.keys[e.id] ? db.keys[e.id].answers : {})
              .filter(q => db.keys[e.id].answers[q] !== null).length,
          }));
      }

      case 'addExam': {
        const subjectId = Number(args[0]);
        const name = String(args[1] || '').trim();
        const questionCount = Number(args[2]) || 30;
        const templateId = String(args[3] || 'omr_' + questionCount);
        if (!name) throw new Error('Enter a name for the exam.');
        if (!db.subjects.some(s => s.id === subjectId)) throw new Error('Pick a subject first.');
        if (db.exams.some(e => e.subjectId === subjectId && e.name.toLowerCase() === name.toLowerCase())) {
          throw new Error('This subject already has an exam with that name.');
        }
        const exam = {
          id: db.nextExam++,
          subjectId,
          name,
          templateId,
          questionCount,
          optionCount: 4,
          createdAt: now(),
        };
        db.exams.push(exam);
        writeFallback(db);
        return { id: exam.id };
      }

      case 'updateExam': {
        const id = Number(args[0]);
        const exam = db.exams.find(e => e.id === id);
        if (!exam) throw new Error('That exam no longer exists.');
        if (args[1] !== null && args[1] !== undefined) exam.name = String(args[1]).trim() || exam.name;
        if (args[2]) {
          const count = Number(args[2]);
          if (count !== exam.questionCount) {
            exam.questionCount = count;
            exam.templateId = args[3] || 'omr_' + count;
            exam.optionCount = 4;
            delete db.keys[id];
          }
        }
        writeFallback(db);
        return { ok: true };
      }

      case 'deleteExam': {
        const id = Number(args[0]);
        db.exams = db.exams.filter(e => e.id !== id);
        delete db.keys[id];
        writeFallback(db);
        return { ok: true };
      }

      case 'saveKey': {
        const examId = Number(args[0]);
        const answers = JSON.parse(args[1] || '[]');
        const marks = JSON.parse(args[2] || '[]');
        if (!db.exams.some(e => e.id === examId)) throw new Error('That exam no longer exists.');
        db.keys[examId] = { answers, marks };
        writeFallback(db);
        return { ok: true };
      }

      case 'loadKey': {
        const examId = Number(args[0]);
        const exam = db.exams.find(e => e.id === examId);
        const count = exam ? exam.questionCount : 0;
        const stored = db.keys[examId] || { answers: [], marks: [] };
        const answers = new Array(count).fill(null);
        const marks = new Array(count).fill(1);
        for (let i = 0; i < count; i++) {
          if (Number.isInteger(stored.answers[i])) answers[i] = stored.answers[i];
          if (Number.isFinite(stored.marks[i])) marks[i] = stored.marks[i];
        }
        return { answers, marks };
      }

      case 'clearKey': {
        delete db.keys[Number(args[0])];
        writeFallback(db);
        return { ok: true };
      }

      default:
        throw new Error('Unknown storage operation: ' + method);
    }
  }

  function invoke(method) {
    const args = Array.prototype.slice.call(arguments, 1);
    if (hasBridge()) return call.apply(null, [method].concat(args));
    return fallback(method, args);
  }

  root.OMRStore = {
    isPersistent: hasBridge,

    listSubjects: () => invoke('listSubjects'),
    addSubject: (name) => invoke('addSubject', name),
    renameSubject: (id, name) => invoke('renameSubject', id, name),
    deleteSubject: (id) => invoke('deleteSubject', id),

    listExams: (subjectId) => invoke('listExams', subjectId),
    addExam: (subjectId, name, questionCount, templateId) =>
      invoke('addExam', subjectId, name, questionCount, templateId),
    updateExam: (id, name, questionCount, templateId) =>
      invoke('updateExam', id, name, questionCount, templateId),
    deleteExam: (id) => invoke('deleteExam', id),

    saveKey: (examId, answers, marks) =>
      invoke('saveKey', examId, JSON.stringify(answers), JSON.stringify(marks)),
    loadKey: (examId) => invoke('loadKey', examId),
    clearKey: (examId) => invoke('clearKey', examId),
  };
})(typeof window !== 'undefined' ? window : globalThis);
