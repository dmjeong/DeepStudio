import assert from "node:assert/strict";
import { join } from "node:path";

export async function verifyExportDraft(page, { temp, fixture, output }) {
  const encryption = page.getByLabel("모델과 설정을 암호화해서 내보내기 (.dvsenc)", { exact: true });
  const key = page.getByLabel("암호키 파일 (.key): 기존 키를 선택하거나 새로 저장할 경로 입력", { exact: true });
  const destination = page.getByLabel("출력 파일의 절대 경로", { exact: true });
  const keyPath = join(temp, "export-test-only.key");
  const encryptedPath = join(temp, "inspection.dvsenc");
  const openExport = () => page.getByRole("button", { name: "내보내기", exact: true }).click();
  const expectEncryptedDraft = async () => {
    await encryption.waitFor();
    assert.equal(await encryption.isChecked(), true);
    assert.equal(await key.inputValue(), keyPath);
    assert.equal(await destination.inputValue(), encryptedPath);
  };

  await openExport();
  await encryption.check();
  await key.fill(keyPath); // Store a path only; no key generation or export.
  await destination.fill(encryptedPath);
  await page.getByRole("button", { name: "프로젝트", exact: true }).click();
  await openExport();
  await expectEncryptedDraft();
  await page.reload();
  await expectEncryptedDraft();

  await encryption.uncheck();
  await page.reload();
  await encryption.waitFor();
  assert.equal(await encryption.isChecked(), false, "explicit uncheck survives remount");
  assert.equal(await key.count(), 0);
  assert.equal(await destination.inputValue(), join(temp, "inspection.onnx"));
  await encryption.check();
  await expectEncryptedDraft();

  const other = await page.request.post("http://127.0.0.1:8766/api/projects", {
    headers: { "X-Studio-Request": "1" },
    data: { name: "export-draft-isolation", task: "classify", parent: temp, class_names: ["ok", "ng"] },
  });
  assert.ok(other.ok(), await other.text());
  await page.reload();
  await encryption.waitFor();
  assert.equal(await encryption.isChecked(), false, "another project does not inherit encryption");
  assert.notEqual(await destination.inputValue(), encryptedPath);
  await encryption.check();
  assert.equal(await key.inputValue(), "", "another project does not inherit a key path");

  const original = await page.request.post("http://127.0.0.1:8766/api/projects/open", {
    headers: { "X-Studio-Request": "1" }, data: { path: fixture.project },
  });
  assert.ok(original.ok(), await original.text());
  await page.reload();
  await expectEncryptedDraft();
  await page.screenshot({ path: join(output, "encrypted-export-draft.png"), fullPage: true });
}
