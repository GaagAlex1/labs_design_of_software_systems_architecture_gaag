pm.test("201 Created", () => {
  pm.response.to.have.status(201);
});

pm.test("Save repo_id from repo_uuid", () => {
  const json = pm.response.json();
  pm.expect(json).to.have.property("repo_uuid");
  pm.environment.set("repo_id", json.repo_uuid);
});