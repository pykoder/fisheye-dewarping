#!/usr/bin/env python3
# coding: utf-8
import subprocess
import re
import sys

def parse_time_output(output):
    """Parse la sortie de /usr/bin/time -v"""
    metrics = {}
    
    patterns = {
        'user_time': r'User time \(seconds\): ([\d.]+)',
        'system_time': r'System time \(seconds\): ([\d.]+)',
        'cpu_percent': r'Percent of CPU this job got: (\d+)%',
        'wall_time': r'Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):([\d.]+)',
        'max_rss_kb': r'Maximum resident set size \(kbytes\): (\d+)',
        'page_faults_minor': r'Minor \(reclaiming a frame\) page faults: (\d+)',
        'page_faults_major': r'Major \(requiring I/O\) page faults: (\d+)',
        'context_switches_vol': r'Voluntary context switches: (\d+)',
        'context_switches_invol': r'Involuntary context switches: (\d+)',
        'exit_status': r'Exit status: (\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            if key == 'wall_time':
                # Parse h:mm:ss or m:ss
                groups = match.groups()
                if groups[0]:  # h:mm:ss
                    hours = int(groups[0])
                    minutes = int(groups[1])
                    seconds = float(groups[2])
                    metrics[key] = hours * 3600 + minutes * 60 + seconds
                else:  # m:ss
                    minutes = int(groups[1])
                    seconds = float(groups[2])
                    metrics[key] = minutes * 60 + seconds
            elif key in ['user_time', 'system_time', 'wall_time']:
                metrics[key] = float(match.group(1))
            elif key == 'cpu_percent':
                metrics[key] = int(match.group(1))
            elif key == 'max_rss_kb':
                metrics[key] = int(match.group(1))
                metrics['max_rss_mb'] = metrics[key] / 1024.0
            else:
                metrics[key] = int(match.group(1))
    
    # Calcul CPU time total et cores utilisés
    if 'user_time' in metrics and 'system_time' in metrics:
        metrics['cpu_time_total'] = metrics['user_time'] + metrics['system_time']
    
    if 'cpu_percent' in metrics:
        metrics['cores_used'] = metrics['cpu_percent'] / 100.0
    
    return metrics

def benchmark_command(cmd):
    """Execute la commande avec /usr/bin/time -v"""
    time_cmd = ['/usr/bin/time', '-v'] + cmd.split()
    
    try:
        result = subprocess.run(
            time_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # /usr/bin/time écrit sur stderr
        metrics = parse_time_output(result.stderr)
        metrics['stdout'] = result.stdout
        
        return metrics
        
    except FileNotFoundError:
        print("❌ Erreur: /usr/bin/time non trouvé sur ce système")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        sys.exit(1)

def print_results(metrics, cmd):
    """Affiche les résultats de manière formatée"""
    print(f"\n🔍 Commande: {cmd}\n")
    print("=" * 70)
    print("📈 RÉSULTATS BENCHMARK")
    print("=" * 70)
    
    if 'wall_time' in metrics:
        print(f"⏱️  Wall time:              {metrics['wall_time']:.2f}s")
    
    if 'cpu_time_total' in metrics:
        print(f"⚙️  CPU time (user+sys):     {metrics['cpu_time_total']:.2f}s")
        print(f"    ├─ User time:           {metrics['user_time']:.2f}s")
        print(f"    └─ System time:         {metrics['system_time']:.2f}s")
    
    if 'cpu_percent' in metrics:
        print(f"🔥 CPU utilization:        {metrics['cpu_percent']}%")
    
    if 'cores_used' in metrics:
        print(f"💻 Cores utilisés:         ~{metrics['cores_used']:.1f}")
    
    if 'max_rss_mb' in metrics:
        print(f"🧠 Mémoire pic:            {metrics['max_rss_mb']:.2f} MB ({metrics['max_rss_kb']} KB)")
    
    if 'page_faults_minor' in metrics:
        print(f"📄 Page faults:            {metrics['page_faults_minor']} minor, {metrics.get('page_faults_major', 0)} major")
    
    if 'context_switches_vol' in metrics:
        print(f"🔄 Context switches:       {metrics['context_switches_vol']} vol, {metrics.get('context_switches_invol', 0)} invol")
    
    if 'exit_status' in metrics:
        status_emoji = "✅" if metrics['exit_status'] == 0 else "❌"
        print(f"{status_emoji} Exit status:            {metrics['exit_status']}")
    
    print("=" * 70)
    
    # Ratio CPU/Wall pour voir l'efficacité du parallélisme
    if 'cpu_time_total' in metrics and 'wall_time' in metrics and metrics['wall_time'] > 0:
        speedup = metrics['cpu_time_total'] / metrics['wall_time']
        print(f"\n💡 Speedup parallèle:      {speedup:.2f}x")
        print(f"   (CPU time / Wall time = {metrics['cpu_time_total']:.2f}s / {metrics['wall_time']:.2f}s)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py <command>")
        print("Example: python3 benchmark.py './script.sh input.mp4'")
        sys.exit(1)
    
    cmd = " ".join(sys.argv[1:])
    
    print(f"🚀 Lancement du benchmark...")
    metrics = benchmark_command(cmd)
    print_results(metrics, cmd)
