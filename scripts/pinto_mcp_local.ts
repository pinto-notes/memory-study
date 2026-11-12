// Pinto Notes' real MCP file tools (search, list, fetch, create, edit, write, ...) and its real
// server instructions, served over HTTP from a local in-process Postgres (PGlite, the repo's own
// test harness). One empty Pinto account per URL path (/conv0, /conv1, ...). Nothing touches
// production: no network, no Supabase.
//
//   deno run -A pinto_mcp_local.ts 8791
//
// Imports the server code from the Pinto repo checkout so the tools are exactly what ships.
const REPO = Deno.env.get("PINTO_REPO") ?? `${Deno.cwd()}/pinto-notes`; // a checkout of github.com/pinto-notes/pinto-notes at 80ce88ca
const mcp = `file://${REPO}/supabase/functions/mcp`;
const { Content } = await import(`${mcp}/tools.ts`); // tools.ts first: shared modules load in its order
const { schemaDB } = await import(`${mcp}/pglite.ts`);
const { account, toolContext } = await import(`${mcp}/sealed.ts`);
const { runFileTool, FILE_TOOLS } = await import(`${mcp}/files_tools.ts`);
const { instructions } = await import(`${mcp}/server.ts`);

type Acct = { pg: unknown; a: unknown };
const accounts = new Map<string, Promise<Acct>>();
function acct(key: string): Promise<Acct> {
  if (!accounts.has(key)) accounts.set(key, (async () => { const pg = await schemaDB(); return { pg, a: await account(pg) }; })());
  return accounts.get(key)!;
}

// Every tool call is logged per account so the study can count what the model did.
const logDir = Deno.env.get("PINTO_LOG_DIR") ?? "runs/mcp-logs";
await Deno.mkdir(logDir, { recursive: true });

const ok = (id: unknown, result: unknown) => ({ jsonrpc: "2.0", id, result });

Deno.serve({ port: Number(Deno.args[0] ?? 8791), hostname: "127.0.0.1", onListen: () => console.error("ready") }, async (req) => {
  const key = new URL(req.url).pathname.replace(/^\/+/, "") || "default";
  if (key === "health") return new Response("ok");
  if (req.method === "GET") return new Response("POST JSON-RPC", { status: 405 });
  if (req.method === "DELETE") return new Response(null, { status: 204 });
  const msg = await req.json();
  const session = req.headers.get("mcp-session-id") ?? crypto.randomUUID();
  const headers = { "content-type": "application/json", "mcp-session-id": session };
  if (msg.id === undefined) return new Response(null, { status: 202, headers });
  const { pg, a } = await acct(key);
  let out: unknown;
  switch (msg.method) {
    case "initialize":
      out = ok(msg.id, { protocolVersion: msg.params?.protocolVersion ?? "2025-06-18",
        capabilities: { tools: { listChanged: false } }, serverInfo: { name: "pinto-notes", version: "local" }, instructions: instructions("files") });
      break;
    case "ping": out = ok(msg.id, {}); break;
    case "tools/list": out = ok(msg.id, { tools: FILE_TOOLS }); break;
    case "tools/call": {
      const name = String(msg.params?.name), args = msg.params?.arguments ?? {};
      // deno-lint-ignore no-explicit-any
      const ctx: any = { ...(await toolContext(pg, a)), session };
      let result: Record<string, unknown>;
      try {
        const r = await runFileTool(name, args, ctx);
        result = r instanceof Content
          ? { content: r.content, ...(r.structured ? { structuredContent: r.structured } : {}) }
          : { content: [{ type: "text", text: typeof r === "string" ? r : JSON.stringify(r, null, 2) }] };
      } catch (e) {
        result = { content: [{ type: "text", text: e instanceof Error ? e.message : String(e) }], isError: true };
      }
      await Deno.writeTextFile(`${logDir}/${key}.jsonl`, JSON.stringify({ t: Date.now(), name, args, isError: !!result.isError }) + "\n", { append: true });
      out = ok(msg.id, result);
      break;
    }
    default: out = { jsonrpc: "2.0", id: msg.id, error: { code: -32601, message: "Method not found" } };
  }
  return new Response(JSON.stringify(out), { headers });
});
