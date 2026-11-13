import express, { Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import { db } from './server/db.js';
import { emailService, isValidEmailFormat } from './server/emailService.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

  app.use(express.json({ limit: '60mb' }));
  app.use(express.urlencoded({ limit: '60mb', extended: true }));

  // Initialize Gemini SDK with User-Agent header as required
  const apiKey = process.env.GEMINI_API_KEY || '';
  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });

  // Health check endpoint
  app.get('/api/health', (req: Request, res: Response) => {
    res.json({
      status: 'ok',
      service: 'NoteFusionAI Backend',
      hasApiKey: Boolean(apiKey),
      timestamp: new Date().toISOString(),
    });
  });

  // JWT Secret & Authentication Middleware
  const JWT_SECRET = process.env.JWT_SECRET || 'notefusion-secure-jwt-academic-secret-key-2026';

  function extractTokenUser(req: Request): { id: string; email: string; name?: string } | null {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
    const token = authHeader.split(' ')[1];
    try {
      return jwt.verify(token, JWT_SECRET) as { id: string; email: string; name?: string };
    } catch {
      return null;
    }
  }

  function requireAuth(req: Request, res: Response, next: () => void) {
    const user = extractTokenUser(req);
    if (!user) {
      res.status(401).json({ error: 'Unauthorized: Authentication required.' });
      return;
    }
    (req as any).user = user;
    next();
  }

  // Model candidate fallback hierarchy for high reliability and quota resilience
  const CANDIDATE_MODELS = [
    'gemini-3.5-flash-lite',
    'gemini-flash-lite-latest',
    'gemini-3.5-flash',
    'gemini-3.6-flash',
    'gemini-3.7-flash',
    'gemini-3.8-flash',
    'gemini-flash-latest',
  ];

  /**
   * Invokes Gemini Interactions API with model fallback
   */
  async function callInteractionsWithFallback(params: {
    input: any[];
    systemInstruction?: string;
  }): Promise<{ model: string; outputText: string; interaction: any }> {
    let lastErr: any = null;
    for (const model of CANDIDATE_MODELS) {
      try {
        console.log(`[Gemini Interactions] Calling model ${model}...`);
        const interaction = await ai.interactions.create({
          model,
          input: params.input,
          system_instruction: params.systemInstruction,
        });

        let outputText = interaction.output_text || '';
        if (!outputText && Array.isArray(interaction.steps)) {
          for (const step of interaction.steps) {
            if (step.type === 'model_output' && Array.isArray(step.content)) {
              for (const c of step.content) {
                if (c.type === 'text' && c.text) {
                  outputText += c.text;
                }
              }
            }
          }
        }

        if (outputText && outputText.trim().length > 0) {
          console.log(`[Gemini Interactions] Successfully received response from ${model} (${outputText.length} chars)`);
          return { model, outputText: outputText.trim(), interaction };
        }
      } catch (err: any) {
        lastErr = err;
        const status = err.status || err.statusCode || 'ERROR';
        console.warn(`[Gemini Interactions] Model ${model} encountered ${status}: ${err.message?.slice(0, 150)}`);
      }
    }
    throw lastErr || new Error('All candidate Gemini models failed to process the Interactions request.');
  }

  /**
   * Invokes Gemini generateContent with model fallback
   */
  async function callGenerateContentWithFallback(params: {
    contents: any;
    config?: any;
  }): Promise<{ model: string; text: string; response: any }> {
    let lastErr: any = null;
    for (const model of CANDIDATE_MODELS) {
      try {
        console.log(`[Gemini GenerateContent] Calling model ${model}...`);
        const response = await ai.models.generateContent({
          model,
          contents: params.contents,
          config: params.config,
        });

        const text = response.text || '';
        if (text && text.trim().length > 0) {
          console.log(`[Gemini GenerateContent] Successfully received response from ${model}`);
          return { model, text: text.trim(), response };
        }
      } catch (err: any) {
        lastErr = err;
        const status = err.status || err.statusCode || 'ERROR';
        console.warn(`[Gemini GenerateContent] Model ${model} encountered ${status}: ${err.message?.slice(0, 150)}`);
      }
    }
    throw lastErr || new Error('All candidate Gemini models failed to process generateContent.');
  }

  /**
   * Safely extracts JSON from model text responses
   */
  function extractJsonFromText(rawText: string): any {
    const trimmed = rawText.trim();
    try {
      return JSON.parse(trimmed);
    } catch (e) {}

    const jsonMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/) || trimmed.match(/(\{[\s\S]*\})/);
    if (jsonMatch) {
      try {
        return JSON.parse(jsonMatch[1].trim());
      } catch (e) {}
    }
    throw new Error('Failed to parse JSON structured output from model response.');
  }

  // Helper to extract YouTube video ID from various formats
  function parseYouTubeVideoId(input: string): string | null {
    if (!input || typeof input !== 'string') return null;
    const trimmed = input.trim();
    const regExp = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?|live|shorts)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
    const match = trimmed.match(regExp);
    if (match && match[1]) return match[1];

    if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed) && !trimmed.includes(' ')) {
      return trimmed;
    }
    return null;
  }

  /**
   * Transcribes spoken dialogue from a public YouTube video using Gemini Multimodal Video Understanding.
   * Prioritizes the official Gemini Interactions API (ai.interactions.create) and falls back to
   * ai.models.generateContent with fileUri for high availability.
   */
  async function transcribeYouTubeVideo(
    resolvedUrl: string,
    videoId: string,
    title?: string
  ): Promise<{ transcript: string; model: string; method: string }> {
    const speechPrompt = `You are NoteFusionAI's expert audio transcriber and speech-to-text engine.
Carefully listen to the entire audio track and spoken dialogue of this video (${title ? `"${title}"` : `YouTube video ${videoId}`}).
Extract the complete, verbatim spoken transcript of everything the speaker(s) say.

Requirements:
1. Provide the complete transcript capturing all spoken dialogue, explanations, and instructions.
2. Group the transcript chronologically with timestamp markers like [00:00], [00:15], etc.
3. If multiple speakers are present, distinguish them where appropriate.
4. Support multilingual speech: accurately transcribe English, Telugu, Hindi, or mixed code-switched speech (e.g., Hinglish/Telugish).
5. Do not invent, summarize, or hallucinate. Transcribe the actual spoken audio.
6. Provide ONLY the timestamped transcript. Do not wrap in conversational chatter or disclaimers.`;

    let lastError: any = null;

    // Phase 1: Gemini Interactions API (Official video-understanding pattern)
    for (const model of CANDIDATE_MODELS) {
      try {
        console.log(`[transcribeYouTubeVideo] Trying Gemini Interactions API with model ${model}...`);
        const interaction = await ai.interactions.create({
          model,
          input: [
            { type: 'video', uri: resolvedUrl },
            { type: 'text', text: speechPrompt },
          ],
        });

        let outputText = interaction.output_text || '';
        if (!outputText && Array.isArray(interaction.steps)) {
          for (const step of interaction.steps) {
            if (step.type === 'model_output' && Array.isArray(step.content)) {
              for (const c of step.content) {
                if (c.type === 'text' && c.text) {
                  outputText += c.text;
                }
              }
            }
          }
        }

        if (outputText && outputText.trim().length > 10) {
          console.log(`[transcribeYouTubeVideo] Interactions API succeeded with ${model} (${outputText.length} chars)`);
          return {
            transcript: outputText.trim(),
            model,
            method: 'gemini-interactions-video',
          };
        }
      } catch (err: any) {
        lastError = err;
        const status = err.status || err.statusCode || 'ERR';
        console.warn(`[transcribeYouTubeVideo] Interactions API on ${model} failed (${status}): ${err.message?.slice(0, 120)}`);
      }
    }

    // Phase 2: Gemini generateContent fallback with native YouTube video/mp4 fileData
    for (const model of CANDIDATE_MODELS) {
      try {
        console.log(`[transcribeYouTubeVideo] Trying Gemini generateContent fallback with model ${model}...`);
        const res = await ai.models.generateContent({
          model,
          contents: [
            { fileData: { fileUri: resolvedUrl, mimeType: 'video/mp4' } },
            { text: speechPrompt },
          ],
        });

        const text = res.text || '';
        if (text && text.trim().length > 10) {
          console.log(`[transcribeYouTubeVideo] generateContent succeeded with ${model} (${text.length} chars)`);
          return {
            transcript: text.trim(),
            model,
            method: 'gemini-generate-content-video',
          };
        }
      } catch (err: any) {
        lastError = err;
        const status = err.status || err.statusCode || 'ERR';
        console.warn(`[transcribeYouTubeVideo] generateContent on ${model} failed (${status}): ${err.message?.slice(0, 120)}`);
      }
    }

    throw lastError || new Error('Unable to extract audio or captions for this YouTube video. The video may be restricted, or Gemini multimodal processing is temporarily unavailable.');
  }

  // Diagnostic YouTube Transcript Extraction Endpoint using Gemini Multimodal Video Understanding
  app.post('/api/extract-youtube', async (req: Request, res: Response) => {
    try {
      const { url } = req.body;
      if (!url) {
        res.status(400).json({
          success: false,
          errorType: 'INVALID_URL',
          message: 'Please provide a YouTube video URL or ID.',
        });
        return;
      }

      const videoId = parseYouTubeVideoId(url);
      if (!videoId) {
        res.status(400).json({
          success: false,
          errorType: 'INVALID_URL',
          message: 'Invalid YouTube link format. Supported formats include youtube.com/watch?v=..., youtu.be/..., or an 11-character video ID.',
        });
        return;
      }

      const resolvedUrl = `https://www.youtube.com/watch?v=${videoId}`;

      // Step 1: Query YouTube oEmbed for official video existence & metadata
      let title = '';
      let author = '';
      let thumbnailUrl = `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
      try {
        const oembedRes = await fetch(
          `https://www.youtube.com/oembed?url=${encodeURIComponent(resolvedUrl)}&format=json`
        );
        if (oembedRes.status === 404) {
          res.json({
            success: false,
            errorType: 'VIDEO_NOT_FOUND',
            videoId,
            message: 'Video Not Found: This YouTube video does not exist or has been deleted.',
          });
          return;
        }
        if (oembedRes.status === 401 || oembedRes.status === 403) {
          res.json({
            success: false,
            errorType: 'VIDEO_PRIVATE',
            videoId,
            message: 'Video Unavailable: This YouTube video is private or embedding is restricted by the creator.',
          });
          return;
        }
        if (oembedRes.ok) {
          const oembedData: any = await oembedRes.json();
          title = oembedData.title || '';
          author = oembedData.author_name || '';
          thumbnailUrl = oembedData.thumbnail_url || thumbnailUrl;
        }
      } catch (e) {
        console.warn('oEmbed fetch notice:', e);
      }

      // Step 2: Use Gemini native YouTube video understanding
      try {
        console.log(`[YouTube Extraction] Transcribing video ${videoId}...`);
        const transcription = await transcribeYouTubeVideo(resolvedUrl, videoId, title);

        res.json({
          success: true,
          videoId,
          title: title || 'YouTube Lecture',
          author: author || 'Instructor',
          thumbnailUrl,
          transcript: transcription.transcript,
          method: transcription.method,
          modelUsed: transcription.model,
        });
        return;
      } catch (aiErr: any) {
        console.error('[YouTube Extraction] Gemini video understanding error:', aiErr?.message);
        res.json({
          success: false,
          errorType: 'EXTRACTION_FAILED',
          videoId,
          title,
          author,
          thumbnailUrl,
          message: aiErr?.message || 'Unable to extract audio or captions for this YouTube video.',
        });
        return;
      }
    } catch (err: any) {
      console.error('Unhandled YouTube extraction error:', err);
      res.status(500).json({
        success: false,
        errorType: 'INTERNAL_ERROR',
        message: err.message || 'An unexpected error occurred during YouTube transcript extraction.',
      });
    }
  });

  // Multimodal Audio, Video URL, or Text Note Generation Endpoint
  app.post('/api/process-notes', async (req: Request, res: Response) => {
    try {
      const {
        audioBase64,
        mimeType,
        rawText,
        youtubeUrl,
        sourceTitle,
        contextHint,
      } = req.body;

      if (!audioBase64 && !rawText && !youtubeUrl) {
        res.status(400).json({ error: 'Please provide either a YouTube URL, audio recording, or text transcript.' });
        return;
      }

      const promptInstruction = `
You are NoteFusionAI, a state-of-the-art academic and professional note-taking artificial intelligence.
Your task is to analyze the provided speech, YouTube lecture video/audio, or lecture/meeting content (which may be in English, Telugu, Hindi, Spanish, or multilingual/code-switched speech like Hinglish/Telugish).

Follow these rules:
1. Multilingual Support: If the spoken content contains non-English words, phrases, or full sentences, transcribe and translate EVERYTHING into coherent, fluent, grammatically correct English while preserving technical terminology and names.
2. Complete Speech Recognition: Provide the complete verbatim chronological transcript with timestamps [MM:SS] in the transcript field.
3. Formulate a comprehensive structured note document in valid JSON format matching this exact schema:
{
  "title": "Clear descriptive title for the lecture or meeting",
  "detectedLanguage": "Original language(s) detected (e.g. 'English & Telugu (Mixed)', 'Hindi & English', 'English')",
  "durationEstimate": "Estimated duration if known or based on content length (e.g. '18 minutes')",
  "transcript": "Complete, polished English transcription of the entire spoken content with speaker flow and timestamps",
  "summary": {
    "executiveSummary": "2-3 well-structured paragraphs providing an in-depth synopsis of the core subject, key dilemmas, mechanisms, and outcomes discussed.",
    "keyTakeaways": [
      "Key point 1 with concrete insight",
      "Key point 2 with concrete insight",
      "Key point 3 with concrete insight",
      "Key point 4 with concrete insight",
      "Key point 5 with concrete insight"
    ],
    "coreThemes": ["Theme 1", "Theme 2", "Theme 3", "Theme 4"]
  },
  "sections": [
    {
      "heading": "Section Topic Heading",
      "content": [
        "In-depth explanation bullet or paragraph 1",
        "In-depth explanation bullet or paragraph 2",
        "Key formulas, system concepts, or methodologies mentioned"
      ],
      "keyTerms": [
        {"term": "Technical Term", "definition": "Precise clear definition"}
      ]
    }
  ],
  "actionItems": [
    {
      "id": "act-1",
      "task": "Concrete, actionable task identified from the discussion",
      "assignee": "Name of assigned person, team, or student (or 'All Attendees' if general)",
      "deadline": "Detected deadline date or relative timeframe (e.g. '2026-10-25', 'Next Friday', 'By end of sprint')",
      "priority": "high",
      "completed": false
    }
  ],
  "calendarEvents": [
    {
      "id": "cal-1",
      "title": "Title for calendar event",
      "description": "Details, agenda, or deliverable needed for this event",
      "startDate": "YYYY-MM-DD formatted date (infer realistic current/upcoming date such as October/November 2026 based on content)",
      "startTime": "HH:MM 24-hr format (e.g. '10:00' or '14:30')",
      "durationMinutes": 60,
      "location": "Classroom, Seminar Hall, or Online Link"
    }
  ],
  "quizQuestions": [
    {
      "question": "Concept check question testing comprehension of the material",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctIndex": 0,
      "explanation": "Brief explanation of why this answer is correct."
    }
  ]
}

Provide ONLY the valid JSON object. Do not wrap in markdown quotes if possible, or use standard JSON.
`;

      let parsedData: any;
      let usedModel = '';

      if (youtubeUrl) {
        const videoId = parseYouTubeVideoId(youtubeUrl);
        const resolvedUrl = videoId ? `https://www.youtube.com/watch?v=${videoId}` : youtubeUrl.trim();
        console.log(`[process-notes] Ingesting YouTube video: ${resolvedUrl}`);

        // Step 1: Transcribe the actual spoken audio from the YouTube video
        const transcription = await transcribeYouTubeVideo(resolvedUrl, videoId || 'video', sourceTitle);

        // Step 2: Synthesize structured notes from the authentic transcript
        console.log(`[process-notes] Synthesizing notes from transcript (${transcription.transcript.length} chars)...`);
        const synthesisPrompt = `${promptInstruction}

YOUTUBE VIDEO INFORMATION:
Title: ${sourceTitle || 'YouTube Lecture'}
URL: ${resolvedUrl}
Additional Context: ${contextHint || 'Educational Lecture / Technical Discourse'}

VERBATIM SPOKEN TRANSCRIPT (WITH TIMESTAMPS):
${transcription.transcript}

Synthesize complete, comprehensive academic study notes based strictly on the spoken transcript above. Return valid JSON.`;

        const noteResult = await callGenerateContentWithFallback({
          contents: synthesisPrompt,
          config: {
            responseMimeType: 'application/json',
            temperature: 0.2,
          },
        });

        usedModel = `${transcription.model} + ${noteResult.model}`;
        parsedData = extractJsonFromText(noteResult.text);
        // Ensure verbatim transcript is preserved
        parsedData.transcript = transcription.transcript;
        parsedData.sourceType = 'youtube';
        parsedData.sourceTitle = sourceTitle || parsedData.title;
        if (videoId) {
          parsedData.videoId = videoId;
          parsedData.thumbnailUrl = `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
        }
      } else if (audioBase64) {
        console.log(`[process-notes] Ingesting audio file/recording (${mimeType || 'audio/webm'})...`);
        const result = await callGenerateContentWithFallback({
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: mimeType || 'audio/webm',
                  data: audioBase64,
                },
              },
              {
                text: `${promptInstruction}\nContext/User Title: ${sourceTitle || 'Untitled Lecture/Meeting'}\nAdditional Context: ${contextHint || 'None'}\n\nIMPORTANT: Return valid JSON.`,
              },
            ],
          },
          config: {
            responseMimeType: 'application/json',
            temperature: 0.2,
          },
        });

        usedModel = result.model;
        parsedData = extractJsonFromText(result.text);
        parsedData.sourceType = 'upload';
        parsedData.sourceTitle = sourceTitle || parsedData.title;
      } else {
        console.log(`[process-notes] Ingesting raw transcript text (${rawText.length} chars)...`);
        const result = await callGenerateContentWithFallback({
          contents: `${promptInstruction}\n\nCONTENT TO ANALYZE:\n${rawText}\n\nContext/User Title: ${sourceTitle || 'Untitled Lecture/Meeting'}\nAdditional Context: ${contextHint || 'None'}\n\nIMPORTANT: Return valid JSON.`,
          config: {
            responseMimeType: 'application/json',
            temperature: 0.2,
          },
        });

        usedModel = result.model;
        parsedData = extractJsonFromText(result.text);
        parsedData.sourceType = 'text';
      }

      // Add unique timestamp & ID
      parsedData.id = 'note-' + Date.now();
      parsedData.timestamp = new Date().toISOString();

      const authUser = extractTokenUser(req);
      if (authUser && authUser.id) {
        parsedData.userId = authUser.id;
        db.saveUserNote(authUser.id, parsedData);
      }

      res.json({ success: true, data: parsedData, modelUsed: usedModel });
    } catch (error: any) {
      console.error('Error generating smart notes:', error);
      res.status(500).json({
        error: error.message || 'Failed to process notes through Gemini AI.',
      });
    }
  });

  // Interactive Notes Q&A Chatbot Endpoint
  app.post('/api/chat-notes', async (req: Request, res: Response) => {
    try {
      const { message, history = [], notesContext } = req.body;

      if (!message) {
        res.status(400).json({ error: 'Message is required.' });
        return;
      }

      const systemInstruction = `
You are the NoteFusionAI Interactive Study & Meeting Assistant.
You have full access to the active lecture/meeting notes and transcript provided below.
Your role:
- Answer student or team questions clearly, accurately, and authoritatively based on the notes and transcript.
- If asked to elaborate on a concept, provide crystal-clear explanations with analogies or code snippets where helpful.
- If asked about deadlines, assignees, or action items, quote the exact detected items.
- If asked for test preparation or flashcards, create helpful practice prompts.
- Maintain an encouraging, intellectual, and professional tone.
- When citing information from the lecture, mention the relevant section or topic.

ACTIVE NOTES CONTEXT:
Title: ${notesContext?.title || 'Lecture / Meeting'}
Detected Language: ${notesContext?.detectedLanguage || 'English'}
Executive Summary: ${notesContext?.summary?.executiveSummary || 'N/A'}
Key Takeaways: ${JSON.stringify(notesContext?.summary?.keyTakeaways || [])}
Sections: ${JSON.stringify(notesContext?.sections || [])}
Action Items: ${JSON.stringify(notesContext?.actionItems || [])}
Calendar Events: ${JSON.stringify(notesContext?.calendarEvents || [])}
Transcript: ${notesContext?.transcript || 'N/A'}
`;

      const contents: any[] = [];

      // Add past history turns if available
      if (Array.isArray(history) && history.length > 0) {
        for (const item of history.slice(-8)) {
          contents.push({
            role: item.role === 'model' ? 'model' : 'user',
            parts: [{ text: item.text }],
          });
        }
      }

      // Append current user prompt
      contents.push({
        role: 'user',
        parts: [{ text: message }],
      });

      const result = await callGenerateContentWithFallback({
        contents,
        config: {
          systemInstruction,
          temperature: 0.4,
        },
      });

      res.json({
        reply: result.text || "I couldn't formulate a response. Please try rephrasing your question.",
        modelUsed: result.model,
      });
    } catch (error: any) {
      console.error('Error in chat notes:', error);
      res.status(500).json({
        error: error.message || 'Failed to generate chat response.',
      });
    }
  });

  // ==========================================
  // AUTHENTICATION & USER MANAGEMENT ENDPOINTS
  // ==========================================

  // Check Email Service Configuration Status
  app.get('/api/auth/email-service-status', (req: Request, res: Response) => {
    const status = emailService.getStatus();
    res.json(status);
  });

  // Developer / Testing helper: Inspect recent verification codes
  app.get('/api/auth/dev-recent-codes', (req: Request, res: Response) => {
    const logs = db.getRecentEmailLogs(10);
    res.json({ logs });
  });

  // Register New User
  app.post('/api/auth/register', async (req: Request, res: Response) => {
    try {
      const { email, password, name } = req.body;

      if (!email || !isValidEmailFormat(email)) {
        res.status(400).json({
          success: false,
          error: 'Please provide a valid, properly formatted email address.',
        });
        return;
      }

      if (!password || typeof password !== 'string' || password.length < 6) {
        res.status(400).json({
          success: false,
          error: 'Password must be at least 6 characters long.',
        });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      const existingUser = db.findUserByEmail(normalizedEmail);

      const otp = Math.floor(100000 + Math.random() * 900000).toString();
      const token = crypto.randomBytes(32).toString('hex');
      const expiresAt = Date.now() + 15 * 60 * 1000; // 15 mins
      const passwordHash = await bcrypt.hash(password, 10);
      const host = req.get('host') || 'localhost:3000';
      const protocol = req.protocol;

      if (existingUser) {
        if (existingUser.is_verified === 1) {
          res.status(409).json({
            success: false,
            error: 'An account with this email address is already registered and verified. Please sign in instead.',
          });
          return;
        }

        // Unverified existing account: update password and send fresh OTP
        db.updateVerificationOtp({
          userId: existingUser.id,
          otp,
          token,
          expiresAt,
        });
        db.updateUserPassword(existingUser.id, passwordHash);

        const emailResult = await emailService.sendVerificationEmail({
          to: normalizedEmail,
          name: name || existingUser.name || 'Scholar',
          otp,
          token,
          reqHost: host,
          protocol,
        });

        res.json({
          success: true,
          requiresVerification: true,
          email: normalizedEmail,
          message: 'Verification code sent to your email inbox. Please verify to activate your account.',
          emailDelivery: emailResult,
        });
        return;
      }

      // Create new user account
      const userId = 'usr_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
      db.createUser({
        id: userId,
        email: normalizedEmail,
        passwordHash,
        name: name ? name.trim() : null,
        verificationOtp: otp,
        verificationToken: token,
        otpExpiresAt: expiresAt,
      });

      const emailResult = await emailService.sendVerificationEmail({
        to: normalizedEmail,
        name: name ? name.trim() : 'Scholar',
        otp,
        token,
        reqHost: host,
        protocol,
      });

      res.status(201).json({
        success: true,
        requiresVerification: true,
        email: normalizedEmail,
        message: 'Account created! A 6-digit verification code has been sent to your email address.',
        emailDelivery: emailResult,
      });
    } catch (err: any) {
      console.error('Registration error:', err);
      res.status(500).json({ success: false, error: 'Registration failed due to a server error.' });
    }
  });

  // Verify Email with 6-Digit OTP
  app.post('/api/auth/verify-otp', async (req: Request, res: Response) => {
    try {
      const { email, otp } = req.body;

      if (!email || !otp) {
        res.status(400).json({ success: false, error: 'Email and verification code are required.' });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      const user = db.findUserByEmail(normalizedEmail);

      if (!user) {
        res.status(404).json({ success: false, error: 'Account not found for this email address.' });
        return;
      }

      if (user.is_verified === 1) {
        // Already verified, generate session token
        const token = jwt.sign(
          { id: user.id, email: user.email, name: user.name },
          JWT_SECRET,
          { expiresIn: '7d' }
        );
        res.json({
          success: true,
          alreadyVerified: true,
          message: 'Account is already verified.',
          token,
          user: { id: user.id, email: user.email, name: user.name, isVerified: true },
        });
        return;
      }

      if (!user.verification_otp || user.verification_otp.trim() !== otp.trim()) {
        res.status(400).json({ success: false, error: 'Invalid verification code. Please check and try again.' });
        return;
      }

      if (user.otp_expires_at && user.otp_expires_at < Date.now()) {
        res.status(400).json({
          success: false,
          error: 'Verification code has expired. Please request a new code.',
        });
        return;
      }

      // Mark verified
      db.markUserVerified(user.id);

      const token = jwt.sign(
        { id: user.id, email: user.email, name: user.name },
        JWT_SECRET,
        { expiresIn: '7d' }
      );

      res.json({
        success: true,
        message: 'Email verified successfully! Welcome to NoteFusionAI.',
        token,
        user: { id: user.id, email: user.email, name: user.name, isVerified: true },
      });
    } catch (err: any) {
      console.error('OTP verification error:', err);
      res.status(500).json({ success: false, error: 'Failed to verify code due to a server error.' });
    }
  });

  // Verify Email via Direct Link (One-Click)
  app.get('/api/auth/verify', async (req: Request, res: Response) => {
    try {
      const token = req.query.token as string;
      if (!token) {
        res.status(400).send('Invalid or missing verification token.');
        return;
      }

      const user = db.findUserByVerificationToken(token);
      if (!user) {
        res.status(404).send(`
          <!DOCTYPE html>
          <html>
          <body style="font-family:sans-serif; text-align:center; padding:50px; background:#F7F1F0; color:#262220;">
            <h1 style="color:#A15C38;">Verification Link Invalid</h1>
            <p>This verification link is invalid or has already been used.</p>
            <a href="/" style="display:inline-block; padding:10px 20px; background:#A15C38; color:#fff; text-decoration:none; border-radius:8px;">Return to NoteFusionAI</a>
          </body>
          </html>
        `);
        return;
      }

      if (user.otp_expires_at && user.otp_expires_at < Date.now()) {
        res.status(400).send(`
          <!DOCTYPE html>
          <html>
          <body style="font-family:sans-serif; text-align:center; padding:50px; background:#F7F1F0; color:#262220;">
            <h1 style="color:#A15C38;">Verification Link Expired</h1>
            <p>This verification link has expired. Please request a new one from the login screen.</p>
            <a href="/" style="display:inline-block; padding:10px 20px; background:#A15C38; color:#fff; text-decoration:none; border-radius:8px;">Return to NoteFusionAI</a>
          </body>
          </html>
        `);
        return;
      }

      db.markUserVerified(user.id);

      res.send(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>Email Verified - NoteFusionAI</title>
          <meta http-equiv="refresh" content="3;url=/?verified=true">
        </head>
        <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; text-align:center; padding:60px 20px; background:#F7F1F0; color:#262220;">
          <div style="max-width:500px; margin:0 auto; background:#fff; padding:40px; border-radius:24px; border:1px solid #E8DFDC; box-shadow:0 4px 20px rgba(0,0,0,0.04);">
            <div style="width:60px; height:60px; background:#A15C38; color:#fff; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:30px; margin-bottom:20px;">✓</div>
            <h1 style="font-size:24px; color:#262220; margin-bottom:12px;">Email Verified Successfully!</h1>
            <p style="color:#6C625E; font-size:14px; line-height:1.6; margin-bottom:24px;">Your NoteFusionAI account is activated. You can now log in and manage your private lecture notes.</p>
            <a href="/?verified=true" style="display:inline-block; padding:12px 28px; background:#A15C38; color:#fff; text-decoration:none; font-weight:600; border-radius:9999px; font-size:14px;">Open NoteFusionAI</a>
            <p style="color:#8C7F7A; font-size:11px; margin-top:20px;">Redirecting automatically in 3 seconds...</p>
          </div>
        </body>
        </html>
      `);
    } catch (err: any) {
      console.error('Link verification error:', err);
      res.status(500).send('Verification failed due to a server error.');
    }
  });

  // Resend OTP
  app.post('/api/auth/resend-otp', async (req: Request, res: Response) => {
    try {
      const { email } = req.body;
      if (!email || !isValidEmailFormat(email)) {
        res.status(400).json({ success: false, error: 'Valid email address is required.' });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      const user = db.findUserByEmail(normalizedEmail);

      if (!user) {
        res.status(404).json({ success: false, error: 'No account found with this email address.' });
        return;
      }

      if (user.is_verified === 1) {
        res.json({
          success: true,
          alreadyVerified: true,
          message: 'Account is already verified. You can log in directly.',
        });
        return;
      }

      const otp = Math.floor(100000 + Math.random() * 900000).toString();
      const token = crypto.randomBytes(32).toString('hex');
      const expiresAt = Date.now() + 15 * 60 * 1000;

      db.updateVerificationOtp({
        userId: user.id,
        otp,
        token,
        expiresAt,
      });

      const host = req.get('host') || 'localhost:3000';
      const emailResult = await emailService.sendVerificationEmail({
        to: normalizedEmail,
        name: user.name || 'Scholar',
        otp,
        token,
        reqHost: host,
        protocol: req.protocol,
      });

      res.json({
        success: true,
        message: 'A new 6-digit verification code has been dispatched to your email address.',
        emailDelivery: emailResult,
      });
    } catch (err: any) {
      console.error('Resend OTP error:', err);
      res.status(500).json({ success: false, error: 'Failed to resend code due to a server error.' });
    }
  });

  // User Login
  app.post('/api/auth/login', async (req: Request, res: Response) => {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        res.status(400).json({ success: false, error: 'Email and password are required.' });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      const user = db.findUserByEmail(normalizedEmail);

      if (!user) {
        res.status(401).json({ success: false, error: 'Invalid email address or password.' });
        return;
      }

      const passwordMatch = await bcrypt.compare(password, user.password_hash);
      if (!passwordMatch) {
        res.status(401).json({ success: false, error: 'Invalid email address or password.' });
        return;
      }

      // Check email verification status
      if (user.is_verified !== 1) {
        // Resend a fresh OTP so the user can verify easily
        const otp = Math.floor(100000 + Math.random() * 900000).toString();
        const token = crypto.randomBytes(32).toString('hex');
        const expiresAt = Date.now() + 15 * 60 * 1000;
        db.updateVerificationOtp({ userId: user.id, otp, token, expiresAt });

        const host = req.get('host') || 'localhost:3000';
        await emailService.sendVerificationEmail({
          to: normalizedEmail,
          name: user.name || 'Scholar',
          otp,
          token,
          reqHost: host,
          protocol: req.protocol,
        });

        res.status(403).json({
          success: false,
          requiresVerification: true,
          email: user.email,
          message: 'Your email address is not yet verified. A fresh verification code has been sent to your inbox.',
        });
        return;
      }

      // Issue JWT session token
      const token = jwt.sign(
        { id: user.id, email: user.email, name: user.name },
        JWT_SECRET,
        { expiresIn: '7d' }
      );

      res.json({
        success: true,
        token,
        user: { id: user.id, email: user.email, name: user.name, isVerified: true },
      });
    } catch (err: any) {
      console.error('Login error:', err);
      res.status(500).json({ success: false, error: 'Login failed due to a server error.' });
    }
  });

  // Current User Profile
  app.get('/api/auth/me', requireAuth, (req: Request, res: Response) => {
    const authUser = (req as any).user;
    const user = db.findUserById(authUser.id);
    if (!user) {
      res.status(404).json({ error: 'User profile not found.' });
      return;
    }

    res.json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        isVerified: user.is_verified === 1,
      },
    });
  });

  // Forgot Password (Send Reset Email)
  app.post('/api/auth/forgot-password', async (req: Request, res: Response) => {
    try {
      const { email } = req.body;
      if (!email || !isValidEmailFormat(email)) {
        res.status(400).json({ success: false, error: 'Valid email address is required.' });
        return;
      }

      const normalizedEmail = email.trim().toLowerCase();
      const user = db.findUserByEmail(normalizedEmail);

      if (user) {
        const token = crypto.randomBytes(32).toString('hex');
        const expiresAt = Date.now() + 60 * 60 * 1000; // 1 hour
        db.setResetPasswordToken(user.id, token, expiresAt);

        const host = req.get('host') || 'localhost:3000';
        await emailService.sendPasswordResetEmail({
          to: normalizedEmail,
          name: user.name || 'Scholar',
          token,
          reqHost: host,
          protocol: req.protocol,
        });
      }

      // Always return success for security (prevents user enumeration)
      res.json({
        success: true,
        message: 'If an account exists with this email address, password reset instructions have been sent.',
      });
    } catch (err: any) {
      console.error('Forgot password error:', err);
      res.status(500).json({ success: false, error: 'Password reset request failed due to a server error.' });
    }
  });

  // Reset Password
  app.post('/api/auth/reset-password', async (req: Request, res: Response) => {
    try {
      const { token, newPassword } = req.body;

      if (!token || !newPassword || typeof newPassword !== 'string' || newPassword.length < 6) {
        res.status(400).json({
          success: false,
          error: 'Valid reset token and password (at least 6 characters) are required.',
        });
        return;
      }

      const user = db.findUserByResetToken(token);
      if (!user) {
        res.status(400).json({ success: false, error: 'Invalid or expired password reset token.' });
        return;
      }

      if (user.reset_expires_at && user.reset_expires_at < Date.now()) {
        res.status(400).json({ success: false, error: 'Password reset token has expired. Please request a new one.' });
        return;
      }

      const passwordHash = await bcrypt.hash(newPassword, 10);
      db.updateUserPassword(user.id, passwordHash);

      res.json({
        success: true,
        message: 'Password reset successful! You can now log in with your new password.',
      });
    } catch (err: any) {
      console.error('Reset password error:', err);
      res.status(500).json({ success: false, error: 'Password reset failed due to a server error.' });
    }
  });

  // ==========================================
  // USER-ISOLATED NOTES MANAGEMENT ENDPOINTS
  // ==========================================

  // Get authenticated user's private saved notes
  app.get('/api/notes', requireAuth, (req: Request, res: Response) => {
    try {
      const userId = (req as any).user.id;
      const notes = db.getUserNotes(userId);
      res.json({ success: true, notes });
    } catch (err: any) {
      console.error('Fetch user notes error:', err);
      res.status(500).json({ success: false, error: 'Failed to retrieve notes.' });
    }
  });

  // Save or update a note for the authenticated user
  app.post('/api/notes', requireAuth, (req: Request, res: Response) => {
    try {
      const userId = (req as any).user.id;
      const noteData = req.body;
      if (!noteData) {
        res.status(400).json({ success: false, error: 'Note data is required.' });
        return;
      }

      const savedNote = db.saveUserNote(userId, noteData);
      res.json({ success: true, note: savedNote });
    } catch (err: any) {
      console.error('Save user note error:', err);
      res.status(500).json({ success: false, error: 'Failed to save note.' });
    }
  });

  // Delete a note for the authenticated user
  app.delete('/api/notes/:id', requireAuth, (req: Request, res: Response) => {
    try {
      const userId = (req as any).user.id;
      const noteId = req.params.id;
      db.deleteUserNote(noteId, userId);
      res.json({ success: true, message: 'Note deleted successfully.' });
    } catch (err: any) {
      console.error('Delete note error:', err);
      res.status(500).json({ success: false, error: 'Failed to delete note.' });
    }
  });

  // Mount Vite middleware in development or static build in production
  if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  } else {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`NoteFusionAI server is running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
