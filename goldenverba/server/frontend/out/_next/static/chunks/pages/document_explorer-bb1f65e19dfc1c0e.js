(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[306],{98404:function(e,t,n){(window.__NEXT_P=window.__NEXT_P||[]).push(["/document_explorer",function(){return n(78006)}])},78006:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return d}});var a=n(85893),s=n(67294),o=n(79123),c=n(85280),l=n(19954);let i=(0,l.getApiHost)();function d(){let[e,t]=(0,s.useState)(""),[n,d]=(0,s.useState)(""),[r,p]=(0,s.useState)("Documentation"),[u,m]=(0,s.useState)("#"),[h,x]=(0,s.useState)([]),[f,_]=(0,s.useState)(null);return(0,s.useEffect)(()=>{let e=async()=>{try{let e=await fetch(i+"/api/get_all_documents"),t=await e.json();console.log(t),x(t.documents)}catch(e){console.error("Failed to fetch all documents:",e)}};e()},[]),(0,s.useEffect)(()=>{let e=async()=>{if(f&&f._additional.id)try{let e=await fetch(i+"/api/get_document",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({document_id:f._additional.id})}),n=await e.json();t(n.document.properties.doc_name),d(n.document.properties.text),p(n.document.properties.doc_type),m(n.document.properties.doc_link)}catch(e){console.error("Failed to fetch document:",e)}};e()},[f]),(0,a.jsx)("main",{className:"flex min-h-screen flex-col items-center justify-between p-12 text-gray-900",children:(0,a.jsxs)("div",{className:"flex flex-col w-full items-start",children:[(0,a.jsxs)("div",{className:"mb-4",children:[(0,a.jsxs)("div",{className:"flex text-lg",children:[(0,a.jsx)("span",{className:"bg-opacity-0 rounded px-2 py-1 hover-container animate-pop-in",children:"The"}),(0,a.jsx)("span",{className:"bg-opacity-0 rounded font-bold px-2 py-1 hover-container animate-pop-in-late",children:"Golden"}),(0,a.jsx)("span",{className:"bg-yellow-200 rounded px-2 py-1 hover-container animate-pop-more-late",children:"RAGtriever"})]}),(0,a.jsx)("h1",{className:"text-8xl font-bold mt-2",children:"Verba"}),(0,a.jsx)("p",{className:"text-sm mt-1 text-gray-400",children:"Retrieval Augmented Generation system powered by Weaviate"})]}),(0,a.jsxs)("div",{className:"flex w-full space-x-4",children:[h.length>0&&(0,a.jsx)("div",{className:"w-1/2 p-2 border-2 shadow-lg h-2/3 border-gray-900 rounded-xl animate-pop-in",children:(0,a.jsx)(c.t7,{height:528,itemCount:h.length,itemSize:100,width:825,children:e=>{let{index:t,style:n}=e;return(0,a.jsx)("button",{style:n,className:" w-full p-4 animate-pop-in-late",onClick:()=>_(h[t]),children:(0,a.jsx)("p",{className:"".concat(l.DOC_TYPE_COLORS[h[t].doc_type]," p-8 w-full rounded-md shadow-md ").concat(l.DOC_TYPE_COLOR_HOVER[h[t].doc_type]),children:h[t].doc_name})},t)}})}),(0,a.jsx)("div",{className:"w-1/2 space-y-4",children:(0,a.jsx)(o.I,{title:e,text:n,extract:"",docLink:u,type:r})})]})]})})}}},function(e){e.O(0,[866,280,172,774,888,179],function(){return e(e.s=98404)}),_N_E=e.O()}]);