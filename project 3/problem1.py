"""
EECE 5614 - Project 3, Problem 1
Q-Learning, SARSA, Tabular Actor-Critic on 20x20 Maze
"""
import numpy as np, matplotlib, os, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# MAZE
# ============================================================
GOAL  = (3, 13)
START = (15, 4)

def init_env():
    E = np.zeros((20,20))
    E[:,0]=1; E[:,19]=1; E[0,:]=1; E[19,:]=1
    E[2,5]=1; E[3,5]=1; E[4,3:17]=1; E[5,3]=1
    E[6,3]=1; E[6,6]=1; E[6,9]=1; E[6,15]=1
    E[7,3]=1; E[7,6]=1; E[7,9]=1; E[7,12:16]=1
    E[8,6]=1; E[8,9]=1; E[8,15]=1; E[9,6]=1; E[9,9]=1; E[9,15]=1
    E[10,1:5]=1; E[10,6]=1; E[10,9:11]=1; E[10,15]=1
    E[11,6]=1; E[11,10]=1; E[11,13]=1; E[11,15:18]=1
    E[12,3:8]=1; E[12,10]=1; E[12,13]=1; E[12,17]=1
    E[13,7]=1; E[13,10]=1; E[13,13]=1; E[13,17]=1
    E[14,7]=1; E[14,10]=1; E[14,13]=1
    E[15,7]=1; E[15,13:17]=1; E[17,1:3]=1; E[17,7:13]=1
    for r,c in [(2,8),(2,16),(4,2),(5,6),(10,18),(14,14),(15,10),(16,10),(17,14),(17,17),(18,7)]:
        E[r,c]=2
    for r,c in [(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),
                (7,2),(7,10),(7,11),(7,17),(8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)]:
        E[r,c]=3
    E[START]=4; E[GOAL]=5
    return E

ACT   = ['up','down','left','right']
DELTA = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}
PERP  = {'up':['left','right'],'down':['left','right'],
         'left':['up','down'],'right':['up','down']}

def step(E, s, action, p):
    """Returns (next_state, reward, is_goal)."""
    perp = PERP[action]
    u = np.random.random()
    if u < 1-p:        actual = action
    elif u < 1-p+p/2:  actual = perp[0]
    else:               actual = perp[1]
    dr,dc = DELTA[actual]
    nr,nc = s[0]+dr, s[1]+dc
    wh = (E[nr,nc]==1)
    ns = s if wh else (nr,nc)
    r = -1.0
    if wh:         r -= 0.8
    if E[ns]==2:   r -= 5
    elif E[ns]==3: r -= 10
    elif E[ns]==5: r += 200
    return ns, r, (ns==GOAL)

def rand_state(E):
    while True:
        r,c = np.random.randint(1,19), np.random.randint(1,19)
        if E[r,c]!=1 and (r,c)!=GOAL: return (r,c)

def eps_greedy(Q, s, eps):
    if np.random.random()<eps: return np.random.randint(4)
    return int(np.argmax(Q[s[0],s[1]]))

def softmax(H, s):
    l = H[s[0],s[1]]; l = l-l.max()
    e = np.exp(l); return e/e.sum()

def greedy_path_ok(QH, E):
    s = START; vis = set()
    for _ in range(500):
        if s==GOAL: return True
        if s in vis: return False
        vis.add(s)
        a = int(np.argmax(QH[s[0],s[1]]))
        dr,dc = DELTA[ACT[a]]
        nr,nc = s[0]+dr,s[1]+dc
        if E[nr,nc]==1: return False
        s = (nr,nc)
    return False

def extract_path(QH, E):
    s=START; path=[s]; vis=set()
    for _ in range(500):
        if s==GOAL: break
        if s in vis: break
        vis.add(s)
        a=int(np.argmax(QH[s[0],s[1]]))
        dr,dc=DELTA[ACT[a]]; nr,nc=s[0]+dr,s[1]+dc
        if E[nr,nc]==1: break
        s=(nr,nc); path.append(s)
    return path

# ============================================================
# ALGORITHMS (with inline first-valid-episode tracking)
# ============================================================

def q_learning(E, p, g, a, eps, nE, mS, seed=None):
    if seed is not None: np.random.seed(seed)
    Q=np.zeros((20,20,4)); rews=[]; fv=nE
    for ep in range(nE):
        s=START; tr=0; n=0
        for _ in range(mS):
            act=eps_greedy(Q,s,eps)
            ns,r,done=step(E,s,ACT[act],p); tr+=r; n+=1
            Q[s[0],s[1],act]+=a*(r+g*np.max(Q[ns[0],ns[1]])-Q[s[0],s[1],act])
            s=ns
            if done: break
        rews.append(tr/n)
        if fv==nE and greedy_path_ok(Q,E): fv=ep+1
    return Q, rews, fv

def sarsa_alg(E, p, g, a, eps, nE, mS, seed=None):
    if seed is not None: np.random.seed(seed)
    Q=np.zeros((20,20,4)); rews=[]; fv=nE
    for ep in range(nE):
        s=START; act=eps_greedy(Q,s,eps); tr=0; n=0
        for _ in range(mS):
            ns,r,done=step(E,s,ACT[act],p); tr+=r; n+=1
            na=eps_greedy(Q,ns,eps)
            Q[s[0],s[1],act]+=a*(r+g*Q[ns[0],ns[1],na]-Q[s[0],s[1],act])
            s=ns; act=na
            if done: break
        rews.append(tr/n)
        if fv==nE and greedy_path_ok(Q,E): fv=ep+1
    return Q, rews, fv

def actor_critic_alg(E, p, g, a, b, nE, mS, seed=None):
    """Per pseudocode: random start, while t<T (no terminal break)."""
    if seed is not None: np.random.seed(seed)
    V=np.zeros((20,20)); H=np.zeros((20,20,4)); rews=[]; fv=nE
    for ep in range(nE):
        s=rand_state(E); tr=0; n=0
        for _ in range(mS):
            pi=softmax(H,s); act=np.random.choice(4,p=pi)
            ns,r,_=step(E,s,ACT[act],p); tr+=r; n+=1
            d=r+g*V[ns[0],ns[1]]-V[s[0],s[1]]
            V[s[0],s[1]]+=a*d
            H[s[0],s[1],act]+=b*d*(1-pi[act])
            s=ns
        rews.append(tr/n)
        if fv==nE and greedy_path_ok(H,E): fv=ep+1
    return H, rews, fv

# ============================================================
# PLOTTING
# ============================================================

def plot_policy(QH, E, title, fn):
    fig,ax=plt.subplots(figsize=(14,14))
    dx={'up':(0,.3),'down':(0,-.3),'left':(-.3,0),'right':(.3,0)}
    clr={0:'white',1:'black',2:'#CC0000',3:'#F5CBA7',4:'#4499FF',5:'#44DD44'}
    for r in range(20):
        for c in range(20):
            ax.add_patch(plt.Rectangle((c,19-r),1,1,facecolor=clr.get(int(E[r,c]),'white'),edgecolor='gray',lw=.5))
            if E[r,c]!=1 and (r,c)!=GOAL:
                a=int(np.argmax(QH[r,c])); ddx,ddy=dx[ACT[a]]
                ax.annotate('',xy=(c+.5+ddx,19-r+.5+ddy),xytext=(c+.5,19-r+.5),
                            arrowprops=dict(arrowstyle='->',color='black',lw=1.2))
    ax.set_xlim(0,20);ax.set_ylim(0,20);ax.set_aspect('equal')
    ax.set_xticks(range(20));ax.set_yticks(range(20));ax.set_yticklabels(range(19,-1,-1))
    ax.set_title(title,fontsize=14);ax.grid(True,alpha=.3)
    plt.tight_layout();plt.savefig(fn,dpi=150,bbox_inches='tight');plt.close()

def plot_path_fig(path, E, title, fn):
    fig,ax=plt.subplots(figsize=(14,14)); ps=set(path)
    clr={0:'white',1:'black',2:'#CC0000',3:'#F5CBA7',4:'#4499FF',5:'#44DD44'}
    for r in range(20):
        for c in range(20):
            fc='#FFCC44' if (r,c) in ps and E[r,c] not in (1,4,5) else clr.get(int(E[r,c]),'white')
            ax.add_patch(plt.Rectangle((c,19-r),1,1,facecolor=fc,edgecolor='gray',lw=.5))
    for i in range(len(path)-1):
        r1,c1=path[i];r2,c2=path[i+1]
        ax.plot([c1+.5,c2+.5],[19-r1+.5,19-r2+.5],'b-',lw=2.5)
    ax.set_xlim(0,20);ax.set_ylim(0,20);ax.set_aspect('equal')
    ax.set_xticks(range(20));ax.set_yticks(range(20));ax.set_yticklabels(range(19,-1,-1))
    ax.set_title(title,fontsize=14);ax.grid(True,alpha=.3)
    plt.tight_layout();plt.savefig(fn,dpi=150,bbox_inches='tight');plt.close()

def plot_curves(all_r, labels, title, fn):
    fig,ax=plt.subplots(figsize=(10,6))
    cc=['blue','red','green','purple']
    for i,(rr,lb) in enumerate(zip(all_r,labels)):
        ca=np.cumsum(rr)/(np.arange(len(rr))+1)
        ax.plot(ca,label=lb,color=cc[i%4],lw=1.5)
    ax.set_xlabel('Episode');ax.set_ylabel('Average Accumulated Reward')
    ax.set_title(title);ax.legend();ax.grid(True,alpha=.3)
    plt.tight_layout();plt.savefig(fn,dpi=150,bbox_inches='tight');plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots'); os.makedirs(D,exist_ok=True)
    E=init_env()
    p,g,a,eps,beta=0.025,0.96,0.25,0.1,0.05
    nE,mS,nR=1000,1000,10

    results={}
    for algo_name, algo_fn, extra in [
        ('Q-Learning', lambda s: q_learning(E,p,g,a,eps,nE,mS,s), {}),
        ('SARSA',      lambda s: sarsa_alg(E,p,g,a,eps,nE,mS,s), {}),
    ]:
        print(f"\n{algo_name}")
        rews_all,found,fvs,bestQH=[],0,[],None
        for run in range(nR):
            QH,rews,fv=algo_fn(run*100+42)
            rews_all.append(rews); fvs.append(fv)
            pf=greedy_path_ok(QH,E)
            if pf: found+=1
            if pf and bestQH is None: bestQH=QH.copy()
            print(f"  Run {run+1}: path={pf}, first_valid={fv}")
        if bestQH is None: bestQH=QH.copy()
        results[algo_name]={'found':found,'fvs':fvs,'avg_r':np.mean(rews_all,axis=0).tolist(),'bestQH':bestQH}
        # Plots
        sn=algo_name.replace('-','').replace(' ','_')
        plot_policy(bestQH,E,f"{algo_name}: Optimal Policy",f"{D}/p1_{sn}_policy.png")
        path=extract_path(bestQH,E)
        plot_path_fig(path,E,f"{algo_name}: Optimal Path ({len(path)-1} steps)",f"{D}/p1_{sn}_path.png")
        plot_curves([np.mean(rews_all,axis=0)],[algo_name],
                    f"{algo_name}: Average Accumulated Reward",f"{D}/p1_{sn}_reward.png")

    # Actor-Critic: default params first (will fail), then tuned
    print("\nACTOR-CRITIC (default β=0.05)")
    ac_found_default=0
    for run in range(nR):
        H,rews,fv=actor_critic_alg(E,p,g,a,beta,nE,mS,run*100+42)
        if greedy_path_ok(H,E): ac_found_default+=1
    print(f"  Default: {ac_found_default}/10 paths (expected 0)")

    # Tuned AC: α=0.1, β=0.5, 2000 episodes, max_steps=200
    print("\nACTOR-CRITIC (tuned: α=0.1, β=0.5, 2000 eps, max_steps=200)")
    ac_rews_all,ac_found,ac_fvs,ac_bestH=[],0,[],None
    for run in range(nR):
        H,rews,fv=actor_critic_alg(E,p,g,0.1,0.5,2000,200,run*100+42)
        ac_rews_all.append(rews[:nE]); ac_fvs.append(min(fv,nE))
        pf=greedy_path_ok(H,E)
        if pf: ac_found+=1
        if pf and ac_bestH is None: ac_bestH=H.copy()
        print(f"  Run {run+1}: path={pf}, first_valid={fv}")
    if ac_bestH is None: ac_bestH=H.copy()
    results['Actor-Critic']={'found':ac_found,'fvs':ac_fvs,
        'avg_r':np.mean(ac_rews_all,axis=0).tolist(),'bestQH':ac_bestH,'found_default':ac_found_default}

    sn='Actor_Critic'
    plot_policy(ac_bestH,E,"Actor-Critic: Optimal Policy (tuned)",f"{D}/p1_{sn}_policy.png")
    ac_path=extract_path(ac_bestH,E)
    plot_path_fig(ac_path,E,f"Actor-Critic: Optimal Path ({len(ac_path)-1} steps)",f"{D}/p1_{sn}_path.png")
    plot_curves([np.mean(ac_rews_all,axis=0)],['Actor-Critic (tuned)'],
                "Actor-Critic: Average Accumulated Reward",f"{D}/p1_{sn}_reward.png")

    # Q4: Combined plot
    plot_curves([results['Q-Learning']['avg_r'], results['SARSA']['avg_r'],
                 results['Actor-Critic']['avg_r']],
                ['Q-Learning','SARSA','Actor-Critic (tuned)'],
                'Problem 1: All Algorithms',f"{D}/p1_all_reward.png")

    # Q6: α sensitivity (Q-Learning)
    print("\nα SENSITIVITY (Q-Learning)")
    alphas=[0.05,0.1,0.25,0.5]; alpha_r={}
    for al in alphas:
        rr=[]
        for run in range(nR):
            _,rews,_=q_learning(E,p,g,al,eps,nE,mS,run*100+42)
            rr.append(rews)
        alpha_r[al]=np.mean(rr,axis=0)
        print(f"  α={al} done")
    fig,ax=plt.subplots(figsize=(10,6))
    for i,al in enumerate(alphas):
        ca=np.cumsum(alpha_r[al])/(np.arange(nE)+1)
        ax.plot(ca,label=f'α={al}',lw=1.5)
    ax.set_xlabel('Episode');ax.set_ylabel('Average Accumulated Reward')
    ax.set_title('Q-Learning: Effect of Learning Rate α');ax.legend();ax.grid(True,alpha=.3)
    plt.tight_layout();plt.savefig(f"{D}/p1_alpha_sensitivity.png",dpi=150,bbox_inches='tight');plt.close()

    # Save JSON
    res_json={
        'ql_found':results['Q-Learning']['found'],
        'sarsa_found':results['SARSA']['found'],
        'ac_found_default':ac_found_default,
        'ac_found_tuned':ac_found,
        'ql_fvs':results['Q-Learning']['fvs'],
        'sarsa_fvs':results['SARSA']['fvs'],
        'ac_fvs':ac_fvs,
        'ac_tuned_params':'α=0.1, β=0.5, 2000 episodes, max_steps=200',
        'ql_path_len':len(extract_path(results['Q-Learning']['bestQH'],E)),
        'sarsa_path_len':len(extract_path(results['SARSA']['bestQH'],E)),
        'ac_path_len':len(ac_path),
    }
    with open(f"{D}/p1_results.json",'w') as f: json.dump(res_json,f,indent=2)

    print("\n=== SUMMARY ===")
    print(f"Q-Learning: {res_json['ql_found']}/10, avg first valid: {np.mean(res_json['ql_fvs']):.0f}")
    print(f"SARSA:      {res_json['sarsa_found']}/10, avg first valid: {np.mean(res_json['sarsa_fvs']):.0f}")
    print(f"AC default: {ac_found_default}/10")
    print(f"AC tuned:   {ac_found}/10, avg first valid: {np.mean([x for x in ac_fvs if x<nE] or [nE]):.0f}")
    print("Problem 1 complete!")

if __name__=='__main__':
    main()
