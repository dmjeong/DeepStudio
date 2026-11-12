# Preserve pytest's exit code, and fail with native stacks on a signal even
# when a crash happens after pytest has already written a passing summary.
set pagination off
set print thread-events off
run
if $_isvoid($_exitcode)
  thread apply all bt 12
  quit 1
end
quit $_exitcode
