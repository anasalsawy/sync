import { appendFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { spawn } from 'node:child_process';
import { GUIAgent, StatusEnum, UITarsModelVersion } from '@ui-tars/sdk';
import { actionParser } from '@ui-tars/action-parser';
import { chromium } from 'playwright';

const goal = 'Research projects and researchers working on AI personas that faithfully reflect real people. Browse academic research, open-source projects, and working systems. Investigate how they represent knowledge, history, voice, and reasoning and verify fidelity. Use the research to design a persona of Sheikh Salah. Establish his identity from evidence and investigate his life, college studies, teachers, expertise, poetry, writings, lectures, and explanatory style. Cite sources and mark uncertainty. Decide how many lobes and knowledge layers are needed and provide the research, plan, and architecture diagram in your final answer.';
function emit(event, details = {}) {
  const record = { time: Date.now()/1000, run: 'sync-ndif-ui-tars', event, ...details };
  appendFileSync(process.env.SYNC_EVENT_LOG, JSON.stringify(record) + '\n');
  console.log(JSON.stringify(record));
}
const bridge = spawn('python', ['ndif_bridge.py'], {stdio:['pipe','pipe','inherit']});
const replies = createInterface({input:bridge.stdout})[Symbol.asyncIterator]();
async function proxyFetch(url, options) {
  const request = new Request(url, options);
  const payload = JSON.parse(await request.text());
  const focused = await page.evaluate(() => {
    const element = document.activeElement;
    return element ? {tag: element.tagName, type: element.getAttribute('type'),
      editable: element.isContentEditable || ['INPUT', 'TEXTAREA'].includes(element.tagName)} : null;
  });
  const editable_fields = await page.evaluate(() => Array.from(document.querySelectorAll(
    'input:not([type=hidden]), textarea, [contenteditable=true]'
  )).flatMap(element => {
    const bounds = element.getBoundingClientRect();
    if (!bounds.width || !bounds.height || bounds.bottom <= 0 || bounds.top >= innerHeight ||
        getComputedStyle(element).visibility === 'hidden') return [];
    const box = [bounds.left, bounds.top, bounds.right, bounds.bottom].map(
      (value, index) => Math.round(1000 * value / (index % 2 ? innerHeight : innerWidth))
    );
    return [{label:element.getAttribute('aria-label') || element.getAttribute('placeholder') ||
      element.getAttribute('name'), box}];
  }));
  payload.browser_observation = {url:page.url(), status:lastStatus, last_action:lastAction,
    error:lastActionError, focused, editable_fields};
  for (let attempt = 1; attempt <= 2; attempt++) {
    bridge.stdin.write(JSON.stringify(payload) + '\n');
    const reply = await replies.next();
    if (reply.done) throw Error('NDIF bridge exited');
    const response = JSON.parse(reply.value);
    if (response.error) {
      return Response.json(response, {status:502});
    }
    const prediction = response.choices?.[0]?.message?.content ?? '';
    let actions = [];
    try {
      actions = actionParser({
        prediction, factor:[1000,1000], uiTarsVersion:UITarsModelVersion.V1_0,
        modelVer:UITarsModelVersion.V1_0, screenContext:page.viewportSize(), scaleFactor:1,
      }).parsed;
    } catch {}
    const supported = ['navigate','click','type','hotkey','scroll','wait','finished','call_user'];
    const onErrorPage = lastStatus >= 400 || /\/home-error\/|\/sorry\//.test(new URL(page.url()).pathname);
    const canLeaveErrorPage = actions.every(action => action.action_type === 'call_user' ||
      (action.action_type === 'navigate' && action.action_inputs?.url &&
        action.action_inputs.url !== page.url() && action.action_inputs.url !== lastAction?.input?.url));
    if ((!onErrorPage || canLeaveErrorPage) && actions.length && actions.every(action => supported.includes(action.action_type))) {
      emit('NDIF decision', {ok:true, attempt});
      return Response.json(response);
    }
    emit('invalid model action', {attempt, responseCharacters:prediction.length});
    payload.action_retry = onErrorPage
      ? 'The browser is on an error or verification page. Choose a different public source or request user input.'
      : 'The previous response could not be executed. Output one supported UI-TARS action.';
  }
  return Response.json({error:{message:'NDIF returned no supported UI-TARS action after retry'}}, {status:422});
}
const browser = await chromium.launch({executablePath:process.env.SYNC_CHROMIUM,proxy:{server:process.env.HTTPS_PROXY}});
const page = await browser.newPage({viewport:{width:1000,height:740}});
let lastStatus, lastAction, lastActionError, finalReport, lastAgentStatus;
page.on('response', response => {
  const request = response.request();
  if (request.isNavigationRequest() && request.frame() === page.mainFrame()) {
    lastStatus = response.status();
  }
});
class BrowserOperator {
  static MANUAL = {ACTION_SPACES:[
    "navigate(url='https://...')", "click(start_box='[x1,y1,x2,y2]')",
    "type(content='text')", "hotkey(key='Enter')",
    "scroll(direction='up or down')", "wait()", "finished(content='final report')", "call_user()"
  ]};
  async screenshot() {
    emit('screenshot', {url:page.url()});
    return {base64:(await page.screenshot()).toString('base64'),scaleFactor:1};
  }
  async execute({parsedPrediction,screenWidth,screenHeight}) {
    const {action_type:action,action_inputs:input}=parsedPrediction;
    emit('action', {action, url:page.url(), input});
    lastAction = {action, input};
    lastActionError = undefined;
    try {
    switch (action) {
      case 'navigate': {
        const url=new URL(input.url);
        if (url.protocol !== 'https:') throw Error('Only public HTTPS browsing is supported');
        lastStatus = (await page.goto(url.href,{waitUntil:'domcontentloaded',timeout:20000}))?.status();
        break;
      }
      case 'click': {
        const [x,y]=input.start_coords;
        await page.mouse.click(x,y);
        break;
      }
      case 'type': {
        const editable = await page.evaluate(() => {
          const element = document.activeElement;
          return element && (element.isContentEditable || ['INPUT', 'TEXTAREA'].includes(element.tagName));
        });
        if (!editable) throw Error('Typing did not run because no editable field has focus. Click a visible input first.');
        await page.keyboard.insertText(input.content);
        break;
      }
      case 'hotkey': await page.keyboard.press(input.key); break;
      case 'scroll': await page.mouse.wheel(0,input.direction==='up'?-600:600); break;
      case 'wait': await page.waitForTimeout(600); break;
      case 'finished':
        finalReport = input.content;
        if (!finalReport?.trim()) throw Error('Finished action requires the research report');
        return {status:StatusEnum.END};
      case 'call_user': return {status:StatusEnum.CALL_USER};
      default: throw Error('Unsupported action: '+action);
    }
    } catch (error) {
      lastActionError = String(error.message);
      emit('action failed', {action, message:lastActionError});
    }
    return {status:StatusEnum.RUNNING};
  }
}
try {
  emit('run started');
  const agent = new GUIAgent({
    model:{baseURL:'http://localhost/v1', apiKey:'local-ndif-bridge',model:'Qwen/Qwen3.5-9B',fetch:proxyFetch,max_tokens:1800},
    logger:{info(){},log(){},warn(){},error(){}},
    operator:new BrowserOperator(), retry:{screenshot:{maxRetries:2}}, uiTarsVersion:UITarsModelVersion.V1_0,
    onData:({data})=>{ lastAgentStatus=data.status; emit('agent status',{status:data.status}); },
    onError:({error})=>{ emit('agent error',{message:String(error?.message ?? error)}); process.exitCode=1; },
  });
  await agent.run(goal);
  emit('run ended',{status:lastAgentStatus, report:finalReport});
  if (lastAgentStatus !== StatusEnum.END || !finalReport?.trim()) process.exitCode=1;
} finally { bridge.kill(); await browser.close(); }
