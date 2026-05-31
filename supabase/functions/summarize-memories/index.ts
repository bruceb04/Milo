import { createClient } from "npm:@supabase/supabase-js@2";

type ConversationRow = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

type DurableMemory = {
  type: "preference" | "project" | "fact" | "instruction";
  content: string;
  confidence: number;
};

const memoryPrompt = `Extract durable memories from the conversation.

Only save information that is likely to be useful in future conversations.
Do not save temporary, sensitive, or trivial facts unless the user explicitly asked to remember them.

Return JSON:
[
  {
    "type": "preference | project | fact | instruction",
    "content": "...",
    "confidence": 0.0 to 1.0
  }
]`;

function jsonResponse(body: Record<string, unknown>, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function getRequiredEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

function vectorLiteral(embedding: number[]): string {
  return JSON.stringify(embedding);
}

function parseMemories(text: string): DurableMemory[] {
  const trimmed = text.trim();
  const jsonStart = trimmed.indexOf("[");
  const jsonEnd = trimmed.lastIndexOf("]");
  const jsonText =
    jsonStart >= 0 && jsonEnd >= jsonStart
      ? trimmed.slice(jsonStart, jsonEnd + 1)
      : trimmed;
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) {
    return [];
  }

  return parsed
    .filter((item) => {
      return (
        item &&
        ["preference", "project", "fact", "instruction"].includes(item.type) &&
        typeof item.content === "string" &&
        typeof item.confidence === "number"
      );
    })
    .map((item) => ({
      type: item.type,
      content: item.content.trim(),
      confidence: Math.max(0, Math.min(1, item.confidence)),
    }))
    .filter((item) => item.content.length > 0 && item.confidence >= 0.5);
}

function getResponseText(response: Record<string, unknown>): string {
  if (typeof response.output_text === "string") {
    return response.output_text;
  }

  const output = Array.isArray(response.output) ? response.output : [];
  const parts: string[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const content = Array.isArray((item as { content?: unknown }).content)
      ? (item as { content: unknown[] }).content
      : [];
    for (const contentItem of content) {
      if (!contentItem || typeof contentItem !== "object") {
        continue;
      }
      const text = (contentItem as { text?: unknown }).text;
      if (typeof text === "string") {
        parts.push(text);
      }
    }
  }

  return parts.join("\n");
}

async function callOpenAI(path: string, body: Record<string, unknown>) {
  const response = await fetch(`https://api.openai.com/v1/${path}`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${getRequiredEnv("OPENAI_API_KEY")}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`OpenAI ${path} failed: ${await response.text()}`);
  }

  return await response.json();
}

async function extractDurableMemories(
  conversation: ConversationRow[],
): Promise<DurableMemory[]> {
  const model = Deno.env.get("MEMORY_SUMMARY_MODEL") ?? "gpt-4.1-mini";
  const transcript = conversation
    .map((row) => `${row.role.toUpperCase()} (${row.created_at}): ${row.content}`)
    .join("\n\n");

  const response = await callOpenAI("responses", {
    model,
    instructions: memoryPrompt,
    input: transcript,
  });

  return parseMemories(getResponseText(response));
}

async function embedMemory(content: string): Promise<number[]> {
  const model = Deno.env.get("EMBEDDING_MODEL") ?? "text-embedding-3-small";
  const dimensions = Number(Deno.env.get("EMBEDDING_DIMENSIONS") ?? "1536");
  const response = await callOpenAI("embeddings", {
    model,
    input: content,
    dimensions,
  });

  return response.data[0].embedding;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  const expectedSecret = Deno.env.get("MEMORY_FUNCTION_SECRET");
  if (expectedSecret && req.headers.get("x-memory-function-secret") !== expectedSecret) {
    return jsonResponse({ error: "Unauthorized" }, 401);
  }

  const body = await req.json();
  const agentId = String(body.agent_id ?? "");
  const userId = String(body.user_id ?? "");
  const conversationId = String(body.conversation_id ?? "");
  const threshold = Number(
    body.threshold ?? Deno.env.get("MEMORY_SUMMARY_THRESHOLD") ?? 20,
  );
  const rawTable = Deno.env.get("PGVECTOR_TABLE") ?? "conversation_memories";

  if (!agentId || !userId || !conversationId) {
    return jsonResponse(
      { error: "agent_id, user_id, and conversation_id are required" },
      400,
    );
  }

  const supabase = createClient(
    getRequiredEnv("SUPABASE_URL"),
    getRequiredEnv("SUPABASE_SERVICE_ROLE_KEY"),
  );

  const { data: agent, error: agentError } = await supabase
    .from("agents")
    .select("id, user_id")
    .eq("id", agentId)
    .eq("user_id", userId)
    .maybeSingle();

  if (agentError) {
    throw agentError;
  }
  if (!agent) {
    return jsonResponse({ error: "Agent not found for user" }, 404);
  }

  const { data: rows, error: rowsError } = await supabase
    .from(rawTable)
    .select("id, role, content, created_at")
    .eq("agent_id", agentId)
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: true });

  if (rowsError) {
    throw rowsError;
  }

  const conversation = (rows ?? []) as ConversationRow[];
  const userQueryCount = conversation.filter((row) => row.role === "user").length;
  if (userQueryCount < threshold) {
    return jsonResponse({
      processed: false,
      user_query_count: userQueryCount,
      threshold,
    });
  }

  const memories = await extractDurableMemories(conversation);
  for (const memory of memories) {
    const embedding = await embedMemory(memory.content);
    const { error } = await supabase.from("user_memories").upsert(
      {
        agent_id: agentId,
        user_id: userId,
        type: memory.type,
        content: memory.content,
        confidence: memory.confidence,
        embedding: vectorLiteral(embedding),
        source_conversation_id: conversationId,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "agent_id,type,content" },
    );
    if (error) {
      throw error;
    }
  }

  const ids = conversation.map((row) => row.id);
  if (ids.length > 0) {
    const { error: deleteError } = await supabase
      .from(rawTable)
      .delete()
      .eq("agent_id", agentId)
      .in("id", ids);
    if (deleteError) {
      throw deleteError;
    }
  }

  return jsonResponse({
    processed: true,
    extracted_count: memories.length,
    deleted_raw_count: ids.length,
  });
});
