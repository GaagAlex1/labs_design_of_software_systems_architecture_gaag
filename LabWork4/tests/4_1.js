pm.test("200 OK", () => {
  pm.response.to.have.status(200);
});

pm.test("status is valid and progress in [0..1]", () => {
  const json = pm.response.json();

  pm.expect(json).to.have.property("status");
  pm.expect(["queued", "running", "succeeded", "failed"]).to.include(json.status);

  pm.expect(json).to.have.property("progress");
  pm.expect(json.progress).to.be.a("number");
  pm.expect(json.progress).to.be.at.least(0);
  pm.expect(json.progress).to.be.at.most(1);
});
