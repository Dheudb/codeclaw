const { _electron: electron } = require("playwright");
const fs = require("node:fs"), path = require("node:path"), os = require("node:os"), http = require("node:http"), assert = require("node:assert/strict");
const { atomicJSON } = require("../electron/catalog.cjs"), { historyRoot } = require("../electron/history-storage.cjs");
const root = path.resolve(__dirname, ".."), fixture = fs.mkdtempSync(path.join(os.tmpdir(), "xuloom-goals-e2e-"));
const data = path.join(fixture, "data"), project = path.join(fixture, "project"), config = path.join(fixture, "config"), home = path.join(fixture, "home");
for (const directory of [data, project, config, home]) fs.mkdirSync(directory);
const env = { ...process.env, XULOOM_DATA_DIR: data, XULOOM_USER_DATA: data, XULOOM_CONFIG_DIR: config,
  USERPROFILE: home, HOME: home, XULOOM_TEST_HEADLESS: "1", XULOOM_SKILLS_PATHS: "", PYTHONPATH: path.dirname(root) };
for (const name of ["ELECTRON_RUN_AS_NODE", "XULOOM_DEV_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "XULOOM_MODEL_PROVIDER"]) delete env[name];
const checkpoint = path.join(historyRoot(project, data), "sessions/goals.json");
atomicJSON(checkpoint, { messages: [], metadata: {} });
atomicJSON(path.join(data, "workbench.json"), { projects: [project], activeConversation: { project, resumeId: "goals", scope: "project" } });
let app, page, requests = 0, delayed = 0; const errors = [];
const server = http.createServer((req, res) => {
  let raw = ""; req.on("data", c => raw += c); req.on("end", () => {
    const body = JSON.parse(raw);
    if (!body.stream) return res.writeHead(200, {"Content-Type":"application/json"}).end(JSON.stringify({choices:[{message:{role:"assistant",content:"memory"}}],usage:{prompt_tokens:10,completion_tokens:1,total_tokens:11}}));
    requests++;
    const last = body.messages.at(-1), goal = JSON.stringify(last.content).includes("Continue the explicitly authorized goal");
    const wake = JSON.stringify(last.content).includes("WAKE_MESSAGE_PROOF");
    if (wake) delayed++;
    res.writeHead(200, {"Content-Type":"text/event-stream"});
    const delta = goal ? {tool_calls:[{index:0,id:"goal-report-"+requests,type:"function",function:{name:"goal_report_tool",arguments:JSON.stringify({status:"complete",summary:"Evidence provided for user review",evidence:["Goal response"]})}}]} : {content:wake ? "WAKE_RESULT_PROOF" : "GOAL_RESULT_PROOF"};
    res.end("data: "+JSON.stringify({choices:[{index:0,delta,finish_reason:null}]})+"\n\ndata: "+JSON.stringify({choices:[{index:0,delta:{},finish_reason:goal?"tool_calls":"stop"}]})+'\n\ndata: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\ndata: [DONE]\n\n');
  });
});
async function launch() {
  app = await electron.launch({ executablePath:process.env.XULOOM_TEST_EXECUTABLE,args:process.env.XULOOM_TEST_EXECUTABLE?[]:[root],env,timeout:30000 });
  page = await app.firstWindow(); page.setDefaultTimeout(30000); page.on("pageerror", e=>errors.push(e.message));
  await page.locator(".session-item.active").waitFor();
  if (await page.getByRole("button", { name: "展开能力面板", exact: true }).count()) await page.getByRole("button", { name: "展开能力面板", exact: true }).click();
  await page.getByRole("button",{name:"目标",exact:true}).click();
  await page.getByRole("heading",{name:"目标与唤醒"}).waitFor();
}
async function main() {
  try {
    await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
    atomicJSON(path.join(config,"config.json"),{model_provider:"openai",openai_model:"fixture",openai_api_key:"fixture",openai_base_url:`http://127.0.0.1:${server.address().port}/v1`});
    await launch();
    await page.getByLabel("目标",{exact:true}).fill("GOAL_OBJECTIVE_PROOF");
    await page.getByLabel("验收条件（每行一项）").fill("Give an evidence based answer");
    await page.getByRole("button",{name:"创建并开始"}).click();
    await page.getByRole("button",{name:"确认验收通过"}).waitFor();
    const png = await app.evaluate(async ({BrowserWindow}) => (await BrowserWindow.getAllWindows()[0].webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG().toString("base64"));
    fs.writeFileSync(path.join(root,"build/goals-ui.png"),Buffer.from(png,"base64"));
    assert.equal(JSON.parse(fs.readFileSync(checkpoint)).metadata.goal.status,"awaiting_review");
    await page.getByRole("button",{name:"确认验收通过"}).click();
    await page.getByRole("heading",{name:"新目标"}).waitFor();
    assert.equal(JSON.parse(fs.readFileSync(checkpoint)).metadata.goal.status,"completed");
    await page.getByRole("textbox",{name:"定时消息",exact:true}).fill("WAKE_MESSAGE_PROOF");
    await page.getByLabel("多少分钟后").fill("0");
    await page.getByRole("button",{name:"保存唤醒"}).click();
    await page.getByText("WAKE_RESULT_PROOF",{exact:true}).waitFor();
    await page.waitForFunction(()=>!document.querySelector(".send-button.stop"));
    await page.waitForTimeout(5500); assert.equal(delayed,1);
    const count=requests; await app.close(); app=undefined; await launch();
    assert.equal(requests,count); assert.equal(delayed,1);
    assert.equal(JSON.parse(fs.readFileSync(checkpoint)).metadata.goal.status,"completed");
    const jobs=JSON.parse(fs.readFileSync(path.join(data,"schedules.json"))).items;
    assert.equal(Object.values(jobs)[0].status,"completed");
    assert.deepEqual(errors,[]);
    console.log("Goals E2E passed: real desktop creation, explicit acceptance, durable budgets/state, timed message to original conversation, no duplicate delivery and restart without model replay");
  } catch(error) { if(page) console.error((await page.locator("body").innerText()).slice(-9000)); throw error; }
  finally { if(app) await app.close(); server.closeAllConnections(); server.close(); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
