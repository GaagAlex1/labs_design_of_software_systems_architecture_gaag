pm.test("422 Unprocessable Entity", () => {
  pm.response.to.have.status(422);
});

pm.test("Body non-empty", () => {
  pm.expect(pm.response.text()).to.not.equal("");
});
