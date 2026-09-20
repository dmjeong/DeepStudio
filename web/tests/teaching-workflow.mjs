import assert from "node:assert/strict";
import { copyFile } from "node:fs/promises";
import { join } from "node:path";

export async function verifyTeaching(page, { temp, fixture, output, base = "http://127.0.0.1:8766" }) {
  const second = join(temp, "second.png");
  await copyFile(fixture.image, second);
  for (const task of ["detect", "segment"]) {
    const response = await page.request.post(`${base}/api/projects`, {
      headers: { "X-Studio-Request": "1" },
      data: { name: `teaching-${task}`, task, parent: temp, class_names: ["background", "part", "defect"] },
    });
    assert.ok(response.ok(), await response.text());
    const project = await response.json();
    await page.reload();
    await page.getByRole("button", { name: "데이터셋", exact: true }).click();
    await page.getByRole("button", { name: "＋ 이미지 추가", exact: true }).click();
    await page.getByLabel("추가할 이미지 파일").setInputFiles([fixture.image, second]);
    await page.getByRole("button", { name: "sample.png 열기", exact: true }).click({ timeout: 30000 });
    const canvas = page.getByLabel("정답 그리기 캔버스");
    await canvas.waitFor();
    await page.getByRole("button", { name: "정답 저장 Ctrl+S", exact: true }).waitFor();
    const point = async (x, y) => { const box = await canvas.boundingBox(); return [box.x+x*box.width, box.y+y*box.height]; };
    const click = async (x, y) => page.mouse.click(...await point(x, y));
    const drag = async (x1, y1, x2, y2) => {
      await page.mouse.move(...await point(x1, y1)); await page.mouse.down();
      await page.mouse.move(...await point(x2, y2), { steps: 5 }); await page.mouse.up();
    };
    await page.getByLabel("정답 클래스", { exact: true }).selectOption("1");
    if (task === "segment") {
      for (const [x, y] of [[.2, .2], [.7, .2], [.7, .7], [.2, .7]]) await click(x, y);
      assert.ok(await page.getByRole("button", { name: "정답 저장 Ctrl+S", exact: true }).isDisabled());
      await page.keyboard.press("Enter");
    } else await drag(.2, .2, .7, .7);
    await page.getByRole("button", { name: "선택/수정 V", exact: true }).click();
    await drag(.7, .7, .8, .8);
    if (task === "segment") {
      await page.getByRole("button", { name: "브러시 B", exact: true }).click();
      await page.getByLabel("정답 클래스", { exact: true }).selectOption("2");
      await page.getByLabel("브러시 px", { exact: true }).fill("8");
      await drag(.4, .5, .6, .5);
      await page.getByRole("button", { name: "지우개 E", exact: true }).click();
      await click(.5, .5);
      await page.getByRole("button", { name: "실행 취소 Ctrl+Z", exact: true }).click();
      assert.equal(await page.locator(".annotation-list > div").count(), 2);
      await page.getByRole("button", { name: "다시 실행", exact: true }).click();
      // An eraser is stored as a background stroke so that it survives a
      // save/reopen round trip, but it is not a foreground object to edit in
      // the object list.  Showing it as one made an erase look like "add".
      assert.equal(await page.locator(".annotation-list > div").count(), 2);
    } else {
      await page.keyboard.press("Control+d");
      assert.equal(await page.locator(".annotation-list > div").count(), 2);
    }
    await page.screenshot({ path: join(output, `teaching-${task}.png`), fullPage: true });
    // A rejected save must keep both edits and the selected image available.
    await page.route("**/api/dataset/edit", async route => route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ detail: "teaching save failure" }) }), { times: 1 });
    await page.getByRole("button", { name: "저장하고 다음 K", exact: true }).click();
    await page.getByRole("alert").filter({ hasText: "teaching save failure" }).first().waitFor();
    assert.match(await page.locator("dialog h2").innerText(), /sample.png/);
    await page.getByRole("button", { name: "저장하고 다음 K", exact: true }).click();
    await page.locator("dialog h2").filter({ hasText: "second.png" }).waitFor({ timeout: 30000 });
    const imagePath = join(project.data.train_dir, "sample.png");
    const endpoint = task === "segment" ? "mask-annotations" : "annotations";
    const saved = await (await page.request.get(`${base}/api/dataset/${endpoint}`, { params: { path: imagePath, split: "train" } })).json();
    assert.equal(task === "segment" ? saved.shapes.length : saved.annotations.length, task === "segment" ? 3 : 2);
    if (task === "segment") {
      assert.equal(saved.persisted, true);
      assert.equal(saved.shapes[0].class_id, 1);
      assert.ok(saved.shapes[0].points[4] > .75, "edited vertex persists");
      assert.equal(saved.shapes[2].class_id, 0, "eraser paints background");
    } else assert.ok(saved.annotations[0].coordinates[2] > .55, "resized box persists");
    // Background/negative image still gets an explicit target when saved.
    await page.getByRole("button", { name: "정답 저장 Ctrl+S", exact: true }).click();
    await page.locator("dialog").waitFor({ state: "detached", timeout: 30000 });
    await page.getByRole("button", { name: "sample.png 열기", exact: true }).click();
    await page.getByLabel("정답 1 클래스", { exact: true }).waitFor();
    assert.equal(await page.locator(".annotation-list > div").count(), task === "segment" ? 3 : 2);
    await page.getByRole("button", { name: "닫기", exact: true }).click();
  }
}
