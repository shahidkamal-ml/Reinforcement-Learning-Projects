"""
EECE 5614 - Project 3, Problem 2
Q-Learning, SARSA, SARSA(λ), Actor-Critic on 4-bit State System
"""
import numpy as np, matplotlib, os, json
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# SYSTEM DEFINITION
# ============================================================
C = np.array([[0,0,-1,0],[1,0,-1,-1],[0,1,0,0],[-1,1,1,0]])
ACTIONS = [np.array([0,0,0,0]), np.array([0,1,0,0]),
           np.array([0,0,1,0]), np.array([0,0,0,1])]
ACT_NAMES = ['a1','a2','a3','a4']
ACT_COST  = [0, 1, 1, 0]   # c(a1)=c(a4)=0, c(a2)=c(a3)=1
NS, NA = 16, 4

def s2v(s): return np.array([(s>>3)&1,(s>>2)&1,(s>>1)&1,s&1])
def v2s(v): return int(v[0])*8+int(v[1])*4+int(v[2])*2+int(v[3])
def bar(x): return (x>0).astype(int)

def step(s, a_idx, p):
    """Transition: s_k = bar(C·s_{k-1}) XOR a_{k-1} XOR n_k. Returns (next_state, reward)."""
    sv = s2v(s)
    noise = (np.random.random(4)<p).astype(int)
    nv = bar(C @ sv) ^ ACTIONS[a_idx] ^ noise
    ns = v2s(nv)
    reward = 5*nv.sum() - ACT_COST[a_idx]
    return ns, reward

def eps_greedy(Q, s, eps):
    if np.random.random()<eps: return np.random.randint(NA)
    return int(np.argmax(Q[s]))

def softmax(H, s):
    l=H[s]-H[s].max(); e=np.exp(l); return e/e.sum()

# ============================================================
# ALGORITHMS (no terminal state — episodes run for max_steps)
# ============================================================

def q_learning(p,g,a,eps,nE,mS,seed=None):
    if seed: np.random.seed(seed)
    Q=np.zeros((NS,NA)); rews=[]
    for ep in range(nE):
        s=np.random.randint(NS); tr=0
        for t in range(mS):
            act=eps_greedy(Q,s,eps)
            ns,r=step(s,act,p); tr+=r
            Q[s,act]+=a*(r+g*np.max(Q[ns])-Q[s,act])
            s=ns
        rews.append(tr/mS)
    return Q, rews

def sarsa_alg(p,g,a,eps,nE,mS,seed=None):
    if seed: np.random.seed(seed)
    Q=np.zeros((NS,NA)); rews=[]
    for ep in range(nE):
        s=np.random.randint(NS); act=eps_greedy(Q,s,eps); tr=0
        for t in range(mS):
            ns,r=step(s,act,p); tr+=r
            na=eps_greedy(Q,ns,eps)
            Q[s,act]+=a*(r+g*Q[ns,na]-Q[s,act])
            s=ns; act=na
        rews.append(tr/mS)
    return Q, rews

def sarsa_lambda(p,g,a,eps,lam,nE,mS,seed=None):
    if seed: np.random.seed(seed)
    Q=np.zeros((NS,NA)); rews=[]
    for ep in range(nE):
        E_tr=np.zeros((NS,NA))
        s=np.random.randint(NS); act=eps_greedy(Q,s,eps); tr=0
        for t in range(mS):
            ns,r=step(s,act,p); tr+=r
            na=eps_greedy(Q,ns,eps)
            d=r+g*Q[ns,na]-Q[s,act]
            E_tr[s,act]+=1
            Q+=a*d*E_tr
            E_tr*=g*lam
            s=ns; act=na
        rews.append(tr/mS)
    return Q, rews

def actor_critic_alg(p,g,a,b,nE,mS,seed=None):
    if seed: np.random.seed(seed)
    V=np.zeros(NS); H=np.zeros((NS,NA)); rews=[]
    for ep in range(nE):
        s=np.random.randint(NS); tr=0
        for t in range(mS):
            pi=softmax(H,s); act=np.random.choice(NA,p=pi)
            ns,r=step(s,act,p); tr+=r
            d=r+g*V[ns]-V[s]
            V[s]+=a*d; H[s,act]+=b*d*(1-pi[act])
            s=ns
        rews.append(tr/mS)
    return H, rews

# ============================================================
# PLOTTING
# ============================================================

def plot_curves(all_r, labels, title, fn):
    fig,ax=plt.subplots(figsize=(10,6))
    cc=['blue','red','green','purple','orange']
    for i,(rr,lb) in enumerate(zip(all_r,labels)):
        ca=np.cumsum(rr)/(np.arange(len(rr))+1)
        ax.plot(ca,label=lb,color=cc[i%5],lw=1.5)
    ax.set_xlabel('Episode');ax.set_ylabel('Average Accumulated Reward')
    ax.set_title(title);ax.legend();ax.grid(True,alpha=.3)
    plt.tight_layout();plt.savefig(fn,dpi=150,bbox_inches='tight');plt.close()

def plot_visitation(counts_dict, fn):
    fig,axes=plt.subplots(2,2,figsize=(14,10)); axes=axes.flatten()
    for idx,(name,counts) in enumerate(counts_dict.items()):
        ax=axes[idx]; labels=[format(s,'04b') for s in range(NS)]
        ax.bar(range(NS),counts,color='steelblue',edgecolor='navy',alpha=.8)
        ax.set_xticks(range(NS));ax.set_xticklabels(labels,rotation=90,fontsize=7)
        ax.set_xlabel('State');ax.set_ylabel('Visits');ax.set_title(name);ax.grid(True,alpha=.3,axis='y')
    plt.suptitle('State Visitation (100 episodes, greedy)',fontsize=14)
    plt.tight_layout();plt.savefig(fn,dpi=150,bbox_inches='tight');plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'plots'); os.makedirs(D,exist_ok=True)
    p,g,a,eps,b,lam=0.1,0.9,0.25,0.15,0.05,0.95
    nE,mS,nR=1000,100,10

    algos = {
        'Q-Learning':    lambda s: q_learning(p,g,a,eps,nE,mS,s),
        'SARSA':         lambda s: sarsa_alg(p,g,a,eps,nE,mS,s),
        'SARSA(λ)':      lambda s: sarsa_lambda(p,g,a,eps,lam,nE,mS,s),
        'Actor-Critic':  lambda s: actor_critic_alg(p,g,a,b,nE,mS,s),
    }

    all_avg, all_pols, all_bestQH = {}, {}, {}
    for name,fn in algos.items():
        print(f"\n{name}")
        rews_runs,pols=[],[]
        bestQH=None
        for run in range(nR):
            QH,rews=fn(run*200+42)
            rews_runs.append(rews)
            pol=[ACT_NAMES[np.argmax(QH[s])] for s in range(NS)]
            pols.append(pol)
            if run==0: bestQH=QH.copy()
            print(f"  Run {run+1}: {pol}")
        all_avg[name]=np.mean(rews_runs,axis=0)
        all_pols[name]=pols
        all_bestQH[name]=bestQH

        sn=name.replace('(','').replace(')','').replace('λ','lambda').replace('-','').replace(' ','_')
        plot_curves([all_avg[name]],[name],f'{name}: Average Accumulated Reward',f"{D}/p2_{sn}_reward.png")

    # Q3: Combined
    plot_curves(list(all_avg.values()),list(all_avg.keys()),
                'Problem 2: All Algorithms',f"{D}/p2_all_reward.png")

    # Q4: State visitation
    print("\nState visitation...")
    vis={}
    for name,QH in all_bestQH.items():
        counts=np.zeros(NS)
        for ep in range(100):
            s=np.random.randint(NS)
            for t in range(mS):
                counts[s]+=1
                act=int(np.argmax(QH[s]))
                s,_=step(s,act,p)
        vis[name]=counts
    plot_visitation(vis,f"{D}/p2_visitation.png")

    # Q5: λ sensitivity
    print("λ sensitivity...")
    lams=[0,0.5,0.95]; lam_r={}
    for l in lams:
        rr=[]
        for run in range(nR):
            _,rews=sarsa_lambda(p,g,a,eps,l,nE,mS,run*200+42)
            rr.append(rews)
        lam_r[l]=np.mean(rr,axis=0)
    plot_curves([lam_r[l] for l in lams],[f'λ={l}' for l in lams],
                'SARSA(λ): Effect of λ',f"{D}/p2_lambda_sensitivity.png")

    # Save
    with open(f"{D}/p2_results.json",'w') as f:
        json.dump({'policies':all_pols,'visitation':{k:v.tolist() for k,v in vis.items()}},f,indent=2,default=str)
    print("\nProblem 2 complete!")

if __name__=='__main__':
    main()
