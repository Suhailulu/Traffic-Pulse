import subprocess

out = subprocess.check_output(['tasklist', '/FO', 'CSV'], text=True)
lines = out.strip().splitlines()
procs = {}
for line in lines[1:]:
    parts = [p.strip('"') for p in line.split('","')]
    if len(parts) >= 5:
        name = parts[0]
        mem_str = parts[4].replace(' K', '').replace(',', '').strip()
        try:
            mem_kb = int(mem_str)
            procs[name] = procs.get(name, 0) + mem_kb
        except ValueError:
            pass

sorted_procs = sorted(procs.items(), key=lambda x: x[1], reverse=True)
print('--- TOP MEMORY CONSUMING PROCESSES (AGGREGATED) ---')
for name, mem in sorted_procs[:15]:
    print(f'{name:<35} {mem/1024:.1f} MB')
