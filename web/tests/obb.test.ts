import assert from "node:assert/strict";
import test from "node:test";
import { obbFromThreePoints } from "../src/obb.ts";

test("비정사각형 이미지에서도 두 변이 직교하고 네 꼭짓점이 유지된다", () => {
  const coords = obbFromThreePoints([.2,.2,.5,.4,.3,.65], 200, 100);
  const p = coords.map((v,i) => v * (i % 2 ? 100 : 200));
  const dot = (p[2]-p[0])*(p[4]-p[2]) + (p[3]-p[1])*(p[5]-p[3]);
  assert.ok(Math.abs(dot) < 1e-8);
  assert.ok(Math.abs(p[0]+p[4]-p[2]-p[6]) < 1e-8);
  assert.ok(coords.every(v => v >= 0 && v <= 1));
});

test("한 줄, 중복 클릭, 이미지 밖 꼭짓점은 저장 가능한 박스로 만들지 않는다", () => {
  assert.throws(() => obbFromThreePoints([.1,.1,.5,.5,.2,.2],100,100));
  assert.throws(() => obbFromThreePoints([.1,.1,.1,.1,.2,.2],100,100));
  assert.throws(() => obbFromThreePoints([.1,.1,.9,.9,.9,.1],100,100));
  assert.throws(() => obbFromThreePoints([.2,.2,.5,.4,.3,.65],0,100));
});
