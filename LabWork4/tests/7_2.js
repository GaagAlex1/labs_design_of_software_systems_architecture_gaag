pm.test("409 Conflict (repo has active jobs)", () => {
  pm.response.to.have.status(409);
});

pm.test("Body non-empty", () => {
  pm.expect(pm.response.text()).to.not.equal("");
});
