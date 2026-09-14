#include <dlfcn.h>
#include <cstdio>
#include "ofxImageEffect.h"
int main(int c, char**v){ void*h=dlopen(v[1],RTLD_NOW); if(!h){printf("dlopen FAIL: %s\\n",dlerror());return 1;}
auto n=(int(*)())dlsym(h,"OfxGetNumberOfPlugins"); auto g=(OfxPlugin*(*)(int))dlsym(h,"OfxGetPlugin");
if(!n||!g){puts("missing exports");return 1;} printf("plugins=%d\\n",n());
for(int i=0;i<n();++i){OfxPlugin*p=g(i); printf("  [%d] api=%s v%d id=%s ver=%u.%u entry=%p\\n",i,p->pluginApi,p->apiVersion,p->pluginIdentifier,p->pluginVersionMajor,p->pluginVersionMinor,(void*)p->mainEntry);} return 0;}
